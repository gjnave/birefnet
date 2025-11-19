import os
import cv2
import numpy as np
import torch
import gradio as gr
import spaces

from glob import glob
from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms

import requests
from io import BytesIO
import zipfile

# Set HF_HOME to a local directory to ensure models are downloaded locally
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")

import transformers


torch.set_float32_matmul_precision('high')
torch.jit.script = lambda f: f

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
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    FG, blur_B = FB_blur_fusion_foreground_estimator_cpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_cpu(image, FG, blur_B, alpha, r=6)[0]


## GPU version refinement
def mean_blur(x, kernel_size):
    """
    equivalent to cv.blur
    x:  [B, C, H, W]
    """
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
    # convert image to float to avoid overflow
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
    # Thanks to the source: https://github.com/ZhengPeng7/BiRefNet/issues/226#issuecomment-3016433728
    FG, blur_B = FB_blur_fusion_foreground_estimator_gpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_gpu(image, FG, blur_B, alpha, r=6)[0]


def refine_foreground(image, mask, r=90, device='cuda'):
    """both image and mask are in range of [0, 1]"""
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
        # Input resolution is on WxH.
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
    # 'Anime-Lite': 'BiRefNet_lite-Anime',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy',
    'General-dynamic': 'BiRefNet_dynamic',
    'Matting-dynamic': 'BiRefNet_dynamic-matting',
}

birefnet = transformers.AutoModelForImageSegmentation.from_pretrained('/'.join(('zhengpeng7', usage_to_weights_file['General'])), trust_remote_code=True)
birefnet.to(device)
birefnet.eval(); birefnet.half()


@spaces.GPU
def predict(images, resolution, weights_file):
    assert (images is not None), 'AssertionError: images cannot be None.'

    global birefnet
    # Load BiRefNet with chosen weights
    _weights_file = '/'.join(('zhengpeng7', usage_to_weights_file[weights_file] if weights_file is not None else usage_to_weights_file['General']))
    print('Using weights: {}.'.format(_weights_file))
    birefnet = transformers.AutoModelForImageSegmentation.from_pretrained(_weights_file, trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval(); birefnet.half()

    try:
        resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    except:
        if weights_file in ['General-HR', 'Matting-HR']:
            resolution = (2048, 2048)
        elif weights_file in ['General-Lite-2K']:
            resolution = (2560, 1440)
        elif weights_file in ['General-reso_512']:
            resolution = (512, 512)
        else:
            if '_dynamic' in weights_file:
                resolution = None
                print('Using the original size (div by 32) for inference.')
            else:
                resolution = (1024, 1024)
        print('Invalid resolution input. Automatically changed to 1024x1024 / 2048x2048 / 2560x1440.')

    if isinstance(images, list):
        # For tab_batch
        save_paths = []
        save_dir = 'preds-BiRefNet'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        tab_is_batch = True
    else:
        images = [images]
        tab_is_batch = False

    for idx_image, image_src in enumerate(images):
        if isinstance(image_src, str):
            if os.path.isfile(image_src):
                image_ori = Image.open(image_src)
            else:
                response = requests.get(image_src)
                image_data = BytesIO(response.content)
                image_ori = Image.open(image_data)
        else:
            image_ori = Image.fromarray(image_src)

        image = image_ori.convert('RGB')
        # Preprocess the image
        if resolution is None:
            resolution_div_by_32 = [int(int(reso)//32*32) for reso in image.size]
            if resolution_div_by_32 != resolution:
                resolution = resolution_div_by_32
        image_preprocessor = ImagePreprocessor(resolution=tuple(resolution))
        image_proc = image_preprocessor.proc(image)
        image_proc = image_proc.unsqueeze(0)

        # Prediction
        with torch.no_grad():
            preds = birefnet(image_proc.to(device).half())[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Show Results
        pred_pil = transforms.ToPILImage()(pred)
        image_masked = refine_foreground(image, pred_pil, device=device)
        image_masked.putalpha(pred_pil.resize(image.size))

        torch.cuda.empty_cache()

        if tab_is_batch:
            save_file_path = os.path.join(save_dir, "{}.png".format(os.path.splitext(os.path.basename(image_src))[0]))
            image_masked.save(save_file_path)
            save_paths.append(save_file_path)

    if tab_is_batch:
        zip_file_path = os.path.join(save_dir, "{}.zip".format(save_dir))
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for file in save_paths:
                zipf.write(file, os.path.basename(file))
        return save_paths, zip_file_path
    else:
        return (image_masked, image_ori)


examples = [[_] for _ in glob('examples/*')][:]
# Add the option of resolution in a text box.
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

descriptions = ('Upload a picture, our model will extract a highly accurate segmentation of the subject in it.\n)'
                 ' The resolution used in our training was `1024x1024`, which is the suggested resolution to obtain good results! `2048x2048` is suggested for BiRefNet_HR.\n'
                 ' Our codes can be found at https://github.com/ZhengPeng7/BiRefNet.\n'
                 ' We also maintain the HF model of BiRefNet at https://huggingface.co/ZhengPeng7/BiRefNet for easier access.')

tab_image = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label='Upload an image'),
        gr.Textbox(lines=1, placeholder="Type the resolution (`WxH`) you want, e.g., `1024x1024`.", label="Resolution"),
        gr.Radio(list(usage_to_weights_file.keys()), value='General', label="Weights", info="Choose the weights you want.")
    ],
    outputs=gr.ImageSlider(label="BiRefNet's prediction", type="pil", format='png'),
    examples=examples,
    api_name="image",
    description=descriptions,
)

tab_text = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Paste an image URL"),
        gr.Textbox(lines=1, placeholder="Type the resolution (`WxH`) you want, e.g., `1024x1024`.", label="Resolution"),
        gr.Radio(list(usage_to_weights_file.keys()), value='General', label="Weights", info="Choose the weights you want.")
    ],
    outputs=gr.ImageSlider(label="BiRefNet's prediction", type="pil", format='png'),
    examples=examples_url,
    api_name="URL",
    description=descriptions+'\nTab-URL is partially modified from https://huggingface.co/spaces/not-lain/background-removal, thanks to this great work!',
)

tab_batch = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload multiple images", type="filepath", file_count="multiple"),
        gr.Textbox(lines=1, placeholder="Type the resolution (`WxH`) you want, e.g., `1024x1024`.", label="Resolution"),
        gr.Radio(list(usage_to_weights_file.keys()), value='General', label="Weights", info="Choose the weights you want.")
    ],
    outputs=[gr.Gallery(label="BiRefNet's predictions"), gr.File(label="Download masked images.")],
    api_name="batch",
    description=descriptions+'\nTab-batch is partially modified from https://huggingface.co/spaces/NegiTurkey/Multi_Birefnetfor_Background_Removal, thanks to this great work!',
)

demo = gr.TabbedInterface(
    [tab_image, tab_text, tab_batch],
    ['image', 'URL', 'batch'],
    title="Official Online Demo of BiRefNet",
)

if __name__ == "__main__":
    demo.launch(debug=True)
