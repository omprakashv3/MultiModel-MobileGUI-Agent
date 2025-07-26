import json
import os
import random
import gc
import math
import psutil
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from typing import List, Dict, Any
import multiprocessing as mp
import signal
from functools import partial

from evaluate_android_control import *
from qwen2vl import Qwen2VL
# from qwen_vl_utils import smart_resize

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _process_image_worker(args):
    try:
        image_root, image_path, resized_width, resized_height = args
        image = Image.open(os.path.join(image_root, image_path))
        if image.size != (resized_width, resized_height):
            image = image.resize((resized_width, resized_height))
        return image
    except Exception as e:
        print(f"Error processing image {args[1] if len(args) > 1 else 'unknown'}: {e}")
        return None

class AndroidControl:
    def __init__(
        self, 
        model_path,
        eval_file, 
        image_root,
        output_dir,
        eval_type,
        thinking=False,
        max_pixels=6400*28*28,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_num_seqs=1,
        seed=42,
        debug=False,
        num_processes=6,
        batch_size=16,  # Made configurable for different hardware
        ):
        self.model_path = model_path
        self.eval_file = eval_file
        self.eval_type = eval_type
        self.max_pixels = max_pixels
        self.image_root = image_root
        self.thinking = thinking
        self.debug = debug
        self.num_processes = num_processes
        self.batch_size = batch_size  # Store as instance variable
        self.output_dir = output_dir
        if thinking:
            self.system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.\nDuring the reasoning process, identify and state the sub-goal of the current step by enclosing it within <step> </step> tags."
        else:
            self.system_prompt = "You are a helpful assistant."
        self.seed = seed

        self.llm = Qwen2VL(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            max_pixels=max_pixels
        )

    def generate_jobs(self) -> List[Dict]:

        with open(self.eval_file, 'r', encoding='utf-8') as f:
            lines = json.load(f)
        
        jobs = []
        for line in tqdm(lines, desc='Generating jobs'):
            # image
            width, height = line['width'], line['height']
            h_bar, w_bar = smart_resize(height, width, max_pixels=self.max_pixels)

            # system message
            tool_system_prompt = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is " + str(w_bar) + "x" + str(h_bar) + ".\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `key`: Perform a key event on the mobile device.\\n    - This supports adb's `keyevent` syntax.\\n    - Examples: \\\"volume_up\\\", \\\"volume_down\\\", \\\"power\\\", \\\"camera\\\", \\\"clear\\\".\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `system_button`: Press the system button.\\n* `open`: Open an app on the device.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"key\", \"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"open\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=key`, `action=type`, and `action=open`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            system_message = self.system_prompt + tool_system_prompt

            # user message
            task_progress = []
            assert len(line["step_check_pams"]) == len(line["step_pams"]) == len(line["episode"]["step_instructions"]) == len(line["episode"]["screenshot_paths"]) - 1
            for i in range(len(line["step_check_pams"])):
                step_check_pam = line["step_check_pams"][i]
                step_pam = line["step_pams"][i]
                step_instruction = line["episode"]["step_instructions"][i]
                screenshot_path = line["episode"]["screenshot_paths"][i]
                if self.eval_type == 'low':
                    user_message_template = "The user query:  {goal}\nCurrent step query: {step_instruction}\nTask progress (You have done the following operation on the current device): {task_progress}"
                    user_message = user_message_template.format(
                        goal=line["episode"]["goal"], 
                        step_instruction=step_instruction, 
                        task_progress=''.join([f'Step {n+1}: {tp}; ' for n, tp in enumerate(task_progress)])
                    )
                elif self.eval_type == 'high':
                    user_message_template = "The user query:  {goal}\nTask progress (You have done the following operation on the current device): {task_progress}"
                    user_message = user_message_template.format(
                        goal=line["episode"]["goal"], 
                        task_progress=''.join([f'Step {n+1}: {tp}; ' for n, tp in enumerate(task_progress)])
                    )
                else:
                    raise ValueError(f"Invalid eval type: {self.eval_type}")
                task_progress.append(json.dumps(step_pam, ensure_ascii=False))
                jobs.append({
                    'question_id': len(jobs),
                    'step_id': i,
                    'episode_id': line['episode']['episode_id'],
                    'messages': [
                        {
                            'role': 'system',
                            'content': system_message
                        },
                        {
                            'role': 'user',
                            'content': user_message + '<image>'
                        }
                    ],
                    'image_path': screenshot_path,
                    'width': width,
                    'height': height,
                    'resized_width': w_bar,
                    'resized_height': h_bar,
                    'check_pams': step_check_pam,
                })
        
        if self.debug:
            random.seed(self.seed)
            jobs = random.sample(jobs, len(jobs)//100)
        return jobs 

    def inference(self, jobs, temperature=0.0, max_tokens=4096, seed=42) -> List[Dict]:
        input_messages = [job['messages'] for job in jobs]
        
        # Memory monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Instead of preloading all images, we'll load them on-demand during batching
        # This prevents memory overflow from loading thousands of images simultaneously
        
        valid_jobs = []
        valid_input_messages = []
        for i, (job, msg) in enumerate(zip(jobs, input_messages)):
            # Just validate that the image file exists
            image_path = os.path.join(self.image_root, job['image_path'])
            if os.path.exists(image_path):
                valid_jobs.append(job)
                valid_input_messages.append(msg)
            else:
                print(f"Skipping job {i} due to missing image: {job['image_path']}")
        
        batch_size = self.batch_size  # Use configurable batch size
        outputs = []
        
        with tqdm(total=len(valid_input_messages), desc='Processing batches') as pbar:
            for i in range(0, len(valid_input_messages), batch_size):
                batch_messages = valid_input_messages[i:i+batch_size]
                batch_jobs = valid_jobs[i:i+batch_size]
                
                # Load images only for current batch
                batch_images = []
                for job in batch_jobs:
                    try:
                        image = Image.open(os.path.join(self.image_root, job['image_path']))
                        if image.size != (job['resized_width'], job['resized_height']):
                            image = image.resize((job['resized_width'], job['resized_height']))
                        batch_images.append(image)
                    except Exception as e:
                        print(f"Error processing image {job['image_path']}: {e}")
                        batch_images.append(None)
                
                # Process only valid images in this batch
                batch_valid_messages = []
                batch_valid_images = []
                batch_valid_jobs = []
                for msg, img, job in zip(batch_messages, batch_images, batch_jobs):
                    if img is not None:
                        batch_valid_messages.append(msg)
                        batch_valid_images.append(img)
                        batch_valid_jobs.append(job)
                
                if batch_valid_messages:
                    batch_outputs = self.llm.chat(batch_valid_messages, batch_valid_images, 
                                                temperature=temperature, 
                                                max_tokens=max_tokens, 
                                                seed=seed)
                    outputs.extend(batch_outputs)
                    
                    # Assign outputs to corresponding jobs
                    for job, output in zip(batch_valid_jobs, batch_outputs):
                        job['llm_output'] = output
                
                # Clear batch images from memory immediately
                for img in batch_images:
                    if img is not None and hasattr(img, 'close'):
                        img.close()
                del batch_images
                
                # Force garbage collection after each batch
                gc.collect()
                
                # Memory monitoring
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                if (i // batch_size) % 10 == 0:  # Print every 10 batches
                    print(f"Batch {i//batch_size + 1}: Memory usage: {current_memory:.2f} MB")
                
                pbar.update(len(batch_messages))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
        
        return [job for job in valid_jobs if 'llm_output' in job]
    
    def compute_scores(self, jobs) -> Dict[str, Any]:
        Type_match_num = 0
        Extact_match_num = 0
        click_match_num = 0
        all_click_num = 0
        error_num = 0
        for job in tqdm(jobs, desc='Computing scores'):
            try:
                output = job['llm_output']
                current_check_pam = job['check_pams']
                pred = output
                
                # Handle thinking mode
                if self.thinking and '</think>' in pred:
                    pred = pred.split('</think>')[-1]
                
                # Handle tool_call format
                if '<tool_call>' in pred:
                    pred = pred.split('<tool_call>')[1]
                else:
                    # More robust parsing for mobile_use format
                    mobile_use_pattern = '{"name": "mobile_use", "arguments":'
                    if mobile_use_pattern in pred:
                        pred = mobile_use_pattern + pred.split(mobile_use_pattern, 1)[1]
                    else:
                        # If the expected pattern is not found, try to find any JSON-like structure
                        import re
                        json_match = re.search(r'\{.*?"arguments":\s*\{.*?\}.*?\}', pred, re.DOTALL)
                        if json_match:
                            pred = json_match.group(0)
                        else:
                            # Last resort: try to find just the arguments part
                            args_match = re.search(r'\{.*?"action".*?\}', pred, re.DOTALL)
                            if args_match:
                                pred = f'{{"name": "mobile_use", "arguments": {args_match.group(0)}}}'
                            else:
                                raise ValueError(f"Could not parse prediction format: {pred[:200]}")
                
                # Handle tool_call closing
                if '</tool_call>' in pred:
                    pred = pred.split('</tool_call>')[0]
                else:
                    if "<conclusion>" in pred:
                        pred = pred.split("<conclusion>")[0]
                    # Clean up malformed JSON
                    pred = pred.rsplit('}}', 1)[0] + '}}'

                # Parse the JSON
                try:
                    parsed = json.loads(pred.strip())
                    if 'arguments' in parsed:
                        pred_action = parsed['arguments']
                    elif 'action' in parsed:
                        # If it's just the action part without the wrapper
                        pred_action = parsed
                    else:
                        raise ValueError(f"No 'arguments' or 'action' found in parsed JSON: {parsed}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON decode error: {e}. Pred: {pred[:200]}")
                
                job['llm_prediction'] = pred_action

                type_match, extact_match = evaluate_android_control_action(pred_action, current_check_pam, job['width'], job['height'], job['resized_width'], job['resized_height'], pred_type ='abs_resized', gt_type='abs_resized')
                if type_match:
                    Type_match_num += 1
                    job['type_match'] = True
                
                if extact_match:
                    Extact_match_num += 1
                    job['extact_match'] = True
                    
                if extact_match and pred_action['action'] == 'click':
                    click_match_num += 1
                    
                if current_check_pam['action'] == 'click':
                    all_click_num += 1
                    
            except Exception as e:
                # Enhanced error logging
                print(f"\n❌ Error processing job {job.get('question_id', 'unknown')}: {str(e)}")
                print(f"Output (first 500 chars): {output[:500] if 'output' in locals() else 'N/A'}")
                if 'pred' in locals():
                    print(f"Parsed pred: {pred[:200]}")
                error_num += 1
                job['parse_error'] = str(e)
                continue
        print('Type_match_num and Extact_match_num: ', Type_match_num, Extact_match_num, '/ all =', len(jobs))
        print('click_match_num:', click_match_num, '/ all =', all_click_num)
        print('error num', error_num)

        res = {
            'type_match_acc': Type_match_num/len(jobs)*100 if len(jobs) > 0 else 0,
            'extact_match_acc': Extact_match_num/len(jobs)*100 if len(jobs) > 0 else 0,
            'click_match_acc': click_match_num/all_click_num*100 if all_click_num > 0 else 0,
            'error_num': error_num,
            'total_jobs': len(jobs),
            'click_jobs': all_click_num,
        }
        
        print(json.dumps(res, indent=' '))

        model_name = self.model_path.split('/')[-1]
        output_dir = os.path.join(self.output_dir, model_name, 'android_control', self.eval_type)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(output_dir, f'scores_{timestamp}{"_debug" if self.debug else ""}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(output_dir, f'jobs_{timestamp}{"_debug" if self.debug else ""}.json'), 'w') as f:
            json.dump(jobs, f, indent=2)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Android Control Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--eval_type', type=str, required=True, choices=['high', 'low'], help='Evaluation type')
    parser.add_argument('--eval_file', type=str, default='./android_control_test.json', help='Path to the evaluation file')
    parser.add_argument('--image_root', type=str, default='./', help='Path to the image root')
    parser.add_argument('--output_dir', type=str, default='./', help='Path to the output directory')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing (reduce if running out of memory)')
    
    args = parser.parse_args()
    
    android_control = AndroidControl(
        model_path=args.model_path, 
        eval_file=args.eval_file,
        image_root=args.image_root,
        output_dir=args.output_dir,
        eval_type=args.eval_type, 
        thinking=args.thinking, 
        debug=args.debug,
        batch_size=args.batch_size,
    )
    jobs = android_control.generate_jobs()
    jobs = android_control.inference(jobs)
    android_control.compute_scores(jobs)
    
