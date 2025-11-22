import os
import cv2
import numpy as np
import torch
import gradio as gr
import tempfile
import shutil

from glob import glob
from typing import Tuple

from PIL import Image
import transformers
from torchvision import transforms

import requests
from io import BytesIO
import zipfile
from tqdm import tqdm


COMMON_RESOLUTIONS = ["256x480", "480x256", "480x480", "480x720", "720x480", "512x512", "1024x1024", "1080x1920", "1920x1080", "1550x2560", "2048x2048", "2560x1440", "Custom"]

torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"


## CPU version refinement
def FB_blur_fusion_foreground_estimator_cpu(image, FG, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FGA = cv2.blur(FG * alpha, (r, r))
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    FG = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG = np.clip(FG, 0, 1)
    return FG, blurred_B


def FB_blur_fusion_foreground_estimator_cpu_2(image, alpha, r=90):
    alpha = alpha[:, :, None]
    FG, blur_B = FB_blur_fusion_foreground_estimator_cpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_cpu(image, FG, blur_B, alpha, r=6)[0]


## GPU version refinement
def mean_blur(x, kernel_size):
    if kernel_size % 2 == 0:
        pad_l = kernel_size // 2 - 1
        pad_r = kernel_size // 2
        pad_t = kernel_size // 2 - 1
        pad_b = kernel_size // 2
    else:
        pad_l = pad_r = pad_t = pad_b = kernel_size // 2

    x_padded = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
    return torch.nn.functional.avg_pool2d(x_padded, kernel_size=(kernel_size, kernel_size), stride=1, count_include_pad=False)


def FB_blur_fusion_foreground_estimator_gpu(image, FG, B, alpha, r=90):
    as_dtype = lambda x, dtype: x.to(dtype) if x.dtype != dtype else x

    input_dtype = image.dtype
    image = as_dtype(image, torch.float32)
    FG = as_dtype(FG, torch.float32)
    B = as_dtype(B, torch.float32)
    alpha = as_dtype(alpha, torch.float32)

    blurred_alpha = mean_blur(alpha, kernel_size=r)

    blurred_FGA = mean_blur(FG * alpha, kernel_size=r)
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = mean_blur(B * (1 - alpha), kernel_size=r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    FG_output = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG_output = torch.clamp(FG_output, 0, 1)

    return as_dtype(FG_output, input_dtype), as_dtype(blurred_B, input_dtype)


def FB_blur_fusion_foreground_estimator_gpu_2(image, alpha, r=90):
    FG, blur_B = FB_blur_fusion_foreground_estimator_gpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_gpu(image, FG, blur_B, alpha, r=6)[0]


def refine_foreground(image, mask, r=90, device='cuda'):
    if mask.size != image.size:
        mask = mask.resize(image.size)

    if device == 'cuda':
        image = transforms.functional.to_tensor(image).float().cuda()
        mask = transforms.functional.to_tensor(mask).float().cuda()
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        estimated_foreground = FB_blur_fusion_foreground_estimator_gpu_2(image, mask, r=r)
        
        estimated_foreground = estimated_foreground.squeeze()
        estimated_foreground = (estimated_foreground.mul(255.0)).to(torch.uint8)
        estimated_foreground = estimated_foreground.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
    else:
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        estimated_foreground = FB_blur_fusion_foreground_estimator_cpu_2(image, mask, r=r)
        estimated_foreground = (estimated_foreground * 255.0).astype(np.uint8)

    estimated_foreground = Image.fromarray(np.ascontiguousarray(estimated_foreground))
    return estimated_foreground


class ImagePreprocessor():
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution[::-1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image: Image.Image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-HR': 'BiRefNet_HR',
    'Matting-HR': 'BiRefNet_HR-matting',
    'Matting': 'BiRefNet-matting',
    'Portrait': 'BiRefNet-portrait',
    'General-reso_512': 'BiRefNet_512x512',
    'General-Lite': 'BiRefNet_lite',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    'Anime-Lite': 'BiRefNet_lite-Anime',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy',
    'General-dynamic': 'BiRefNet_dynamic',
}

# Initialize model
birefnet = transformers.AutoModelForImageSegmentation.from_pretrained(
    '/'.join(('zhengpeng7', usage_to_weights_file['General'])), 
    trust_remote_code=True
)
birefnet.to(device)
birefnet.eval()
birefnet.half()


def get_resolution(resolution_dropdown, resolution_custom, weights_file):
    """Parse resolution from dropdown or custom input"""
    if resolution_dropdown == "Custom":
        resolution_str = resolution_custom
    else:
        resolution_str = resolution_dropdown

    try:
        resolution = [int(int(reso)//32*32) for reso in resolution_str.strip().split('x')]
    except:
        if weights_file in ['General-HR', 'Matting-HR']:
            resolution = (2048, 2048)
        elif weights_file in ['General-Lite-2K']:
            resolution = (2560, 1440)
        elif weights_file in ['General-reso_512']:
            resolution = (512, 512)
        else:
            if weights_file in ['General-dynamic']:
                resolution = None
            else:
                resolution = (1024, 1024)
    return resolution


def load_model(weights_file):
    """Load BiRefNet model with specified weights"""
    global birefnet
    _weights_file = '/'.join(('zhengpeng7', usage_to_weights_file[weights_file] if weights_file else usage_to_weights_file['General']))
    print(f'Loading weights: {_weights_file}')
    birefnet = transformers.AutoModelForImageSegmentation.from_pretrained(_weights_file, trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval()
    birefnet.half()


def process_single_image(image, resolution):
    """Process a single image and return masked result"""
    global birefnet
    
    image_ori = image if isinstance(image, Image.Image) else Image.fromarray(image)
    image_rgb = image_ori.convert('RGB')
    
    # Handle dynamic resolution
    if resolution is None:
        resolution = [int(int(reso)//32*32) for reso in image_rgb.size]
    
    # Preprocess
    image_preprocessor = ImagePreprocessor(resolution=tuple(resolution))
    image_proc = image_preprocessor.proc(image_rgb)
    image_proc = image_proc.unsqueeze(0)

    # Inference
    with torch.no_grad():
        preds = birefnet(image_proc.to(device).half())[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Post-process
    pred_pil = transforms.ToPILImage()(pred)
    image_masked = refine_foreground(image_rgb, pred_pil, device=device)
    image_masked.putalpha(pred_pil.resize(image_rgb.size))

    return image_masked, image_ori


def predict(images, resolution_dropdown, resolution_custom, weights_file):
    """Main prediction function for images"""
    assert images is not None, 'AssertionError: images cannot be None.'

    resolution = get_resolution(resolution_dropdown, resolution_custom, weights_file)
    load_model(weights_file)

    if isinstance(images, list):
        # Batch processing
        save_paths = []
        save_dir = 'preds-BiRefNet'
        os.makedirs(save_dir, exist_ok=True)
        
        for image_src in images:
            if isinstance(image_src, str):
                if os.path.isfile(image_src):
                    image_ori = Image.open(image_src)
                else:
                    response = requests.get(image_src)
                    image_ori = Image.open(BytesIO(response.content))
            else:
                image_ori = Image.fromarray(image_src)

            image_masked, _ = process_single_image(image_ori, resolution)
            torch.cuda.empty_cache()

            save_file_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_src))[0]}.png")
            image_masked.save(save_file_path)
            save_paths.append(save_file_path)

        # Create zip
        zip_file_path = os.path.join(save_dir, f"{save_dir}.zip")
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for file in save_paths:
                zipf.write(file, os.path.basename(file))
        return save_paths, zip_file_path
    else:
        # Single image
        if isinstance(images, str):
            if os.path.isfile(images):
                image_ori = Image.open(images)
            else:
                response = requests.get(images)
                image_ori = Image.open(BytesIO(response.content))
        else:
            image_ori = Image.fromarray(images)

        image_masked, image_ori = process_single_image(image_ori, resolution)
        torch.cuda.empty_cache()
        return (image_masked, image_ori)


def process_video(video_path, resolution_dropdown, resolution_custom, weights_file, 
                  output_format, bg_color, progress=gr.Progress()):
    """Process video frame by frame"""
    assert video_path is not None, 'Please upload a video.'

    resolution = get_resolution(resolution_dropdown, resolution_custom, weights_file)
    load_model(weights_file)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    
    # Determine output settings based on format
    if output_format == "webm (VP9 + transparency)":
        output_ext = ".webm"
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        has_alpha = True
    elif output_format == "mp4 (H.264, no transparency)":
        output_ext = ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        has_alpha = False
    else:  # PNG sequence
        output_ext = ".zip"
        has_alpha = True

    # Parse background color
    if bg_color == "Transparent":
        bg = None
    elif bg_color == "Green":
        bg = (0, 255, 0)
    elif bg_color == "Black":
        bg = (0, 0, 0)
    elif bg_color == "White":
        bg = (255, 255, 255)
    else:
        bg = None

    frames_output = []
    
    progress(0, desc="Processing video frames...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Process frame
        image_masked, _ = process_single_image(image, resolution)
        
        # Apply background if needed
        if bg is not None and not has_alpha:
            # Composite onto background
            bg_image = Image.new('RGBA', image_masked.size, bg + (255,))
            composite = Image.alpha_composite(bg_image, image_masked)
            frame_result = np.array(composite.convert('RGB'))
        elif has_alpha:
            frame_result = np.array(image_masked)
        else:
            frame_result = np.array(image_masked.convert('RGB'))

        frames_output.append(frame_result)
        
        # Clear VRAM periodically
        if frame_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        progress((frame_idx + 1) / frame_count, desc=f"Processing frame {frame_idx + 1}/{frame_count}")
        frame_idx += 1

    cap.release()

    # Write output
    if output_format == "PNG sequence (zip)":
        # Save as PNG sequence in zip
        png_dir = os.path.join(temp_dir, "frames")
        os.makedirs(png_dir, exist_ok=True)
        
        for i, frame in enumerate(frames_output):
            frame_path = os.path.join(png_dir, f"frame_{i:06d}.png")
            Image.fromarray(frame).save(frame_path)
        
        output_path = os.path.join(temp_dir, "output_frames.zip")
        with zipfile.ZipFile(output_path, 'w') as zipf:
            for f in sorted(os.listdir(png_dir)):
                zipf.write(os.path.join(png_dir, f), f)
    else:
        # Write video file
        output_path = os.path.join(temp_dir, f"output{output_ext}")
        
        if output_format == "webm (VP9 + transparency)":
            # Use ffmpeg for webm with alpha
            png_dir = os.path.join(temp_dir, "frames")
            os.makedirs(png_dir, exist_ok=True)
            
            for i, frame in enumerate(frames_output):
                frame_path = os.path.join(png_dir, f"frame_{i:06d}.png")
                Image.fromarray(frame).save(frame_path)
            
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(png_dir, 'frame_%06d.png'),
                '-c:v', 'libvpx-vp9',
                '-pix_fmt', 'yuva420p',
                '-b:v', '2M',
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        else:
            # MP4 output
            out_height, out_width = frames_output[0].shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            for frame in frames_output:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()

    # Extract and merge audio if present
    try:
        import subprocess
        # Check if video has audio
        probe_cmd = ['ffprobe', '-i', video_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        if result.stdout:  # Has audio
            audio_path = os.path.join(temp_dir, "audio.aac")
            final_output = os.path.join(temp_dir, f"final_output{output_ext}")
            
            # Extract audio
            subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'copy', audio_path], 
                         capture_output=True, check=True)
            
            # Merge audio with video
            subprocess.run(['ffmpeg', '-y', '-i', output_path, '-i', audio_path, 
                          '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_output],
                         capture_output=True, check=True)
            
            output_path = final_output
    except Exception as e:
        print(f"Audio processing skipped: {e}")

    progress(1.0, desc="Complete!")
    
    return output_path, output_path


# Examples setup
examples = [[_] for _ in glob('examples/*')][:]
for idx_example, example in enumerate(examples):
    if 'My_' in example[0]:
        example_resolution = '2048x2048'
        model_choice = 'Matting-HR'
    else:
        example_resolution = '1024x1024'
        model_choice = 'General'
    examples[idx_example] = examples[idx_example] + [example_resolution, model_choice]

examples_url = [
    ['https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg'],
]
for idx_example_url, example_url in enumerate(examples_url):
    examples_url[idx_example_url] = examples_url[idx_example_url] + ['1024x1024', 'General']

descriptions = """Upload a picture or video, and the model will extract a highly accurate segmentation of the subject.

**Recommended resolutions:** `1024x1024` for standard models, `2048x2048` for BiRefNet_HR models.

- GitHub: https://github.com/ZhengPeng7/BiRefNet
- HuggingFace: https://huggingface.co/ZhengPeng7/BiRefNet
"""


# Build Gradio Interface
with gr.Blocks(title="BiRefNet - Background Removal") as demo:
    gr.Markdown(
        "<h1 align='center'>BiRefNet - Background Removal</h1>"
        "<h3 align='center'>Image & Video Processing</h3>"
    )

    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(label='Upload an image')
            image_output = gr.Image(label="Result", type="pil", format='png')
        with gr.Row():
            resolution_dropdown = gr.Dropdown(
                choices=COMMON_RESOLUTIONS, 
                value="1024x1024", 
                label="Resolution Preset"
            )
            resolution_custom = gr.Textbox(
                lines=1, 
                placeholder="WxH, e.g., 1024x1024", 
                label="Custom Resolution", 
                visible=False
            )
            weights_radio = gr.Radio(
                list(usage_to_weights_file.keys()), 
                value='General', 
                label="Model Weights"
            )
        
        resolution_dropdown.change(
            lambda v: gr.update(visible=v == "Custom"),
            inputs=resolution_dropdown,
            outputs=resolution_custom,
        )
        
        image_button = gr.Button("Remove Background", variant="primary")
        image_button.click(
            fn=predict,
            inputs=[image_input, resolution_dropdown, resolution_custom, weights_radio],
            outputs=[image_output],
        )
        gr.Examples(examples, inputs=[image_input, resolution_dropdown, weights_radio], fn=predict, outputs=image_output)

    with gr.Tab("Video"):
        gr.Markdown("### Video Background Removal")
        gr.Markdown("⚠️ **Note:** Video processing can be slow. For best results, use shorter clips or lower resolutions.")
        
        with gr.Row():
            video_input = gr.Video(label='Upload a video')
            video_output = gr.Video(label="Processed Video")
        
        with gr.Row():
            video_file_output = gr.File(label="Download Result")
        
        with gr.Row():
            resolution_dropdown_video = gr.Dropdown(
                choices=COMMON_RESOLUTIONS, 
                value="512x512", 
                label="Resolution Preset",
                info="Lower resolution = faster processing"
            )
            resolution_custom_video = gr.Textbox(
                lines=1, 
                placeholder="WxH, e.g., 512x512", 
                label="Custom Resolution", 
                visible=False
            )
        
        with gr.Row():
            weights_radio_video = gr.Radio(
                list(usage_to_weights_file.keys()), 
                value='General-Lite', 
                label="Model Weights",
                info="Lite models are faster for video"
            )
        
        with gr.Row():
            output_format = gr.Dropdown(
                choices=["webm (VP9 + transparency)", "mp4 (H.264, no transparency)", "PNG sequence (zip)"],
                value="webm (VP9 + transparency)",
                label="Output Format"
            )
            bg_color = gr.Dropdown(
                choices=["Transparent", "Green", "Black", "White"],
                value="Transparent",
                label="Background Color",
                info="For formats without transparency support"
            )

        resolution_dropdown_video.change(
            lambda v: gr.update(visible=v == "Custom"),
            inputs=resolution_dropdown_video,
            outputs=resolution_custom_video,
        )

        video_button = gr.Button("Process Video", variant="primary")
        video_button.click(
            fn=process_video,
            inputs=[video_input, resolution_dropdown_video, resolution_custom_video, 
                   weights_radio_video, output_format, bg_color],
            outputs=[video_output, video_file_output],
        )

    with gr.Tab("URL"):
        url_input = gr.Textbox(label="Paste an image URL")
        url_output = gr.Image(label="Result", type="pil", format='png')
        with gr.Row():
            resolution_dropdown_url = gr.Dropdown(
                choices=COMMON_RESOLUTIONS, 
                value="1024x1024", 
                label="Resolution Preset"
            )
            resolution_custom_url = gr.Textbox(
                lines=1, 
                placeholder="WxH, e.g., 1024x1024", 
                label="Custom Resolution", 
                visible=False
            )
            weights_radio_url = gr.Radio(
                list(usage_to_weights_file.keys()), 
                value='General', 
                label="Model Weights"
            )

        resolution_dropdown_url.change(
            lambda v: gr.update(visible=v == "Custom"),
            inputs=resolution_dropdown_url,
            outputs=resolution_custom_url,
        )

        url_button = gr.Button("Remove Background", variant="primary")
        url_button.click(
            fn=predict,
            inputs=[url_input, resolution_dropdown_url, resolution_custom_url, weights_radio_url],
            outputs=[url_output],
        )
        gr.Examples(examples_url, inputs=[url_input, resolution_dropdown_url, weights_radio_url], fn=predict, outputs=url_output)

    with gr.Tab("Batch"):
        batch_input = gr.File(
            label="Upload multiple images", 
            type="filepath", 
            file_count="multiple"
        )
        batch_gallery = gr.Gallery(label="Results")
        batch_file_output = gr.File(label="Download All (ZIP)")
        
        with gr.Row():
            resolution_dropdown_batch = gr.Dropdown(
                choices=COMMON_RESOLUTIONS, 
                value="1024x1024", 
                label="Resolution Preset"
            )
            resolution_custom_batch = gr.Textbox(
                lines=1, 
                placeholder="WxH, e.g., 1024x1024", 
                label="Custom Resolution", 
                visible=False
            )
            weights_radio_batch = gr.Radio(
                list(usage_to_weights_file.keys()), 
                value='General', 
                label="Model Weights"
            )
        
        resolution_dropdown_batch.change(
            lambda v: gr.update(visible=v == "Custom"),
            inputs=resolution_dropdown_batch,
            outputs=resolution_custom_batch,
        )

        batch_button = gr.Button("Process Batch", variant="primary")
        batch_button.click(
            fn=predict,
            inputs=[batch_input, resolution_dropdown_batch, resolution_custom_batch, weights_radio_batch],
            outputs=[batch_gallery, batch_file_output],
        )

    gr.Markdown(descriptions)


if __name__ == "__main__":
    demo.launch(debug=True)
