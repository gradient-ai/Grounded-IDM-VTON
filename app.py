import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import numpy as np
# Grounding DINO

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything.segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download

import gradio as gr
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import spaces
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

sam_checkpoint = 'sam_vit_h_4b8939.pth'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))


# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
  return annotated_frame, boxes 


def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
  

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def detect_clothing(img, has_hat, has_gloves, human_img):
    print(type(img))
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = img.convert("RGB").resize((768,1024))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    if has_hat:
        hat = ', hat'
    if has_gloves:
        gloves = ', gloves'
    if has_hat == False:
        hat = ''
    if has_gloves == False:
        gloves = ''
    
    annotated_frame, detected_boxes = detect(image_transformed, image, text_prompt=f"clothing clothes tops bottoms{hat}{gloves}", model=groundingdino_model)
    
    segmented_frame_masks = segment(image, sam_predictor, boxes=detected_boxes)

    
    mask = segmented_frame_masks[0][0].cpu().numpy()

    count = 1
    for i in range(len(segmented_frame_masks)+1):
        try:
            mask += segmented_frame_masks[count][0].cpu().numpy()
            print(f'added {count} total mask(s) to initial')
            count += 1
            
        except:
            None
    print(f'the mask from grounded is {type(mask)}')
    keypoints = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))
    mask2, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask2 = mask2.resize((768,1024))
    print(f'the mask from og is {type(mask2)}')
    image_mask_pil = Image.fromarray(np.asarray(mask2)-mask)
    # image_mask_pil = Image.fromarray(mask)

    return image_mask_pil


@spaces.GPU
def start_tryon(dict, garm_img,garment_des,is_checked,is_checked_crop,use_grounding,has_hat, has_gloves, denoise_steps,seed):
    device = "cuda"
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")    
    print('check 1')
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))
    print('check x')

    if is_checked:
        print('automasking...')
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    elif use_grounding:
        mask = detect_clothing(dict["background"], has_hat, has_gloves, human_img)
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
    # return images[0], mask_gray

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=False)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)
            with gr.Row():
                use_grounding = gr.Checkbox(label = 'Yes', info = 'Use Grounded Segment Anything to generate mask (better than auto-masking)', value = True)
            with gr.Row():
                has_hat = gr.Checkbox(label = 'Yes', info = 'Look for a hat to mask in the outfit', value = False)
                has_gloves = gr.Checkbox(label = 'Yes', info = 'Look for a gloves to mask in the outfit', value = False)
            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)




    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=20, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)



    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop,use_grounding, has_hat, has_gloves, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')

            


image_blocks.launch(share = True)