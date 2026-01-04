import os
import torch
from diffusers import KandinskyPipeline, KandinskyPriorPipeline, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPModel, AutoImageProcessor
from transformers import pipeline as pipeline_caption
from PIL import Image
import numpy as np
import torchvision
import imageio
from torchvision.transforms.functional import to_tensor
import json
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def my_flat(x):
    return x.view(x.size(0), -1).detach().cpu()
        
def get_sd_v14_model(device):
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    pipeline.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    unet = pipeline.unet.to(device).eval().half()
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device).half()
    vae = pipeline.vae.to(device).eval().half()
    return unet, tokenizer, text_encoder, vae, noise_scheduler

def numpy_chunk(arr, num_chunks, axis=0):
    """
    Splits a NumPy array into approximately equal chunks along the specified axis.
    
    Parameters:
    - arr: NumPy array to be split
    - num_chunks: Number of chunks to create
    - axis: Axis along which to split the array
    
    Returns:
    - List of NumPy arrays
    """
    # Calculate the size of each chunk
    chunk_size = arr.shape[axis] // num_chunks
    remainder = arr.shape[axis] % num_chunks
    chunks = []
    
    # Determine indices to split the array
    indices = np.cumsum([0] + [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)])
    
    # Use np.split to split the array at the calculated indices
    splits = np.split(arr, indices[1:-1], axis=axis)
    
    return splits

def my_resize(img_t, siz):
    """
    Resize and crop an image tensor to the specified size.
    
    Parameters:
    - img_t (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
    - siz (int): Target size for the cropped square (output dimensions will be siz x siz).
    
    Returns:
    - torch.Tensor: Resized and cropped tensor of shape (C, siz, siz) or (B, C, siz, siz).
    """
    
    img_t = torchvision.transforms.Resize(siz+3)(img_t)
    
    # Manually calculate center crop to siz x siz
    start_x = (img_t.size(-1) - siz) // 2
    start_y = (img_t.size(-2) - siz) // 2

    if img_t.dim() == 3:  # CHW format
        img_t = img_t[:, start_y:start_y + siz, start_x:start_x + siz]
    elif img_t.dim() == 4:  # BCHW format
        img_t = img_t[:, :, start_y:start_y + siz, start_x:start_x + siz]
    else:
        raise ValueError("Unsupported tensor shape: {}".format(img_t.shape))

    return img_t

def load_and_convert_image_batch(image_paths, device):
    """
    Loads a batch of images from file paths and returns them as a list of tensors.

    Parameters:
    - image_paths (list of str): List of image file paths.
    - device (torch.device): Device to which the tensors should be moved (e.g., 'cpu' or 'cuda').

    Returns:
    - list of torch.Tensor: List of tensors, each of shape (H, W, C), dtype `uint8`, with values in the range [0, 255].
    """
    return [torch.tensor(imageio.imread(path, pilmode='RGB')) for path in image_paths]

def preprocess_image_batch(batch_img_t, siz, device):
    """
    Preprocesses a batch of image tensors for input to a neural network.

    Parameters:
    - batch_img_t (list of torch.Tensor): List of tensors, each of shape (H, W, C), with values in the range [0, 255].
    - siz (int, optional): Target size for resizing and cropping. Default is 512.

    Returns:
    - torch.Tensor: Preprocessed tensor of shape (B, 3, siz, siz), dtype `float16`, with values scaled to the range [-1, 1].
    """
    batch_img_t = [img.permute(2, 0, 1).to(device).half() for img in batch_img_t]
    batch_img_t = [my_resize(img, siz) for img in batch_img_t]
    batch_img_t = torch.stack(batch_img_t)
    # Scale dynamic range [0, 255] -> [-1, 1]
    batch_img_t = 2 * (batch_img_t / 255.0) - 1
    return batch_img_t

def postprocess_image(img_t, siz, do_resize=True):
    if do_resize:
        img_t = my_resize(img_t, siz)
    img_t = (img_t / 2 + 0.5).clamp(0, 1) * 255
    img_t = img_t.detach().cpu()
    img_t = img_t.permute(0, 2, 3, 1).float().numpy()
    img_t = img_t.round().astype("uint8")  # [0]
    return img_t

def normalize_batch(batch, epsilon=1e-8):
    """normalize each element in a pytorch batch (dim 0 is the batch dimension)"""
    # Normalize this tensor without assuming its element dimensionality
    dims_to_normalize = tuple(range(1, batch.dim()))  # Create a tuple of dimensions excluding the batch dimension
    # normalize each batch of noise by its norm
    norms = torch.norm(batch, p=2, dim=dims_to_normalize, keepdim=True)  # calculate the L2 norm per-element-in-batch 
    return  batch / (norms + epsilon)

# ===========================
# Drop-in replacement: SDv1.4 PDE/Stein residual criterion
# (keep your helpers; only replace factory_sdv14_based_criterion with this)
# ===========================

def factory_sdv14_based_criterion(
    device,
    num_noise,
    epsilon_reg,
    time_frac,
    tokenizer,
    text_encoder,
    image_to_text,
    vae,
    scheduler,
    unet,
    clip,          # unused now (kept for signature compatibility)
    processor,     # unused now (kept for signature compatibility)
    cos,           # unused now (kept for signature compatibility)
    siz,
    prompts_list=None,
    return_terms=False,
):
    """
    New method: PDE/Stein residual on latent diffusion score (no training).
    Statistic per image:
        r(z_t, t) = div(f)(constant) + <f(z_t), s_theta(z_t,t)>
    where f = ∇g, g(z)=||Δ z||^2 (Δ: 2D Laplacian per-channel conv), hence
        f(z) = 2 Δ^2 z,   div(f)=trace(Hess g)= 40 * (C*H*W)  for Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]].
    Score from SD noise-prediction:
        z_t = sqrt(alpha_bar) z_0 + sigma_t * eps
        eps_pred = UNet(z_t,t,cond)
        s_theta(z_t,t) = - eps_pred / sigma_t
    """

    # --- Laplacian conv kernel (depthwise, per-channel) ---
    lap = torch.tensor([[0.0, 1.0, 0.0],
                        [1.0,-4.0, 1.0],
                        [0.0, 1.0, 0.0]], device=device, dtype=torch.float32).view(1,1,3,3)

    def laplacian(x):  # x: (B,C,H,W)
        B,C,H,W = x.shape
        k = lap.repeat(C, 1, 1, 1)  # (C,1,3,3) depthwise
        return F.conv2d(x, k, padding=1, groups=C)

    def sdv14_pde_criterion(images_raw):
        num_images = len(images_raw)

        # --------- preprocess to [-1,1] BCHW ---------
        images = preprocess_image_batch(images_raw, siz, device)  # (B,3,siz,siz), half, [-1,1]

        # --------- captions/prompts (same as your current code) ---------
        if prompts_list is not None:
            prompts = (prompts_list * num_images) if len(prompts_list) == 1 else prompts_list
            assert len(prompts) == num_images
        else:
            prompts = []
            for cur_image_raw in images_raw:
                image_use = TF.to_pil_image(cur_image_raw.permute(2, 0, 1))
                prompt = "Generate a caption for the image that contains only facts and detailed."
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_use},
                        {"type": "text", "text": prompt},
                    ],
                }]
                out = image_to_text(text=messages, generate_kwargs={"max_new_tokens": 76})
                des = out[0]["generated_text"][-1]["content"].strip()
                prompts.append(des)

        # --------- text embeddings (SD uses max_length=77) ---------
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_noise)

        text_tokens = tokenizer(
            expanded_prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_tokens.input_ids.to(device)
        with torch.no_grad():
            text_emb = text_encoder(input_ids).last_hidden_state  # (B*num_noise,77,dim)

        # --------- encode images -> latents z0 ---------
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample()  # (B,4,64,64) for 512
            z0 = z0 * vae.config.scaling_factor

        # repeat per noise
        z0 = z0.repeat_interleave(num_noise, dim=0).half()  # (B*num_noise,4,64,64)

        # --------- sample eps + build z_t ---------
        eps = torch.randn_like(z0, device=device).half()
        eps = normalize_batch(eps, epsilon_reg).half()
        sqrt_d = torch.prod(torch.tensor(z0.shape[1:], device=device)).float().sqrt()
        eps = eps * sqrt_d

        # choose timestep
        t = int(time_frac * scheduler.config.num_train_timesteps)
        t = torch.full((z0.shape[0],), t, device=device, dtype=torch.long)

        # z_t from scheduler helper (consistent with SD)
        zt = scheduler.add_noise(original_samples=z0, noise=eps, timesteps=t).half()

        # --------- UNet predicts eps_pred ---------
        with torch.no_grad():
            eps_pred = unet(zt, t, encoder_hidden_states=text_emb)[0].half()

        # --------- score from eps_pred ---------
        # sigma_t = sqrt(1 - alpha_bar_t)
        alpha_bar = scheduler.alphas_cumprod[t[0].item()].to(device).float()
        sigma_t = torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8)
        score = -(eps_pred.float() / sigma_t).half()  # (B*num_noise,4,64,64)

        # --------- f = ∇g, g(z)=||Δz||^2 -> f(z)=2 Δ^2 z ---------
        zt_f = zt.float()  # conv in fp32 for stability
        lap1 = laplacian(zt_f)
        lap2 = laplacian(lap1)          # Δ^2 z
        f_field = (2.0 * lap2).half()   # (B*num_noise,4,64,64)

        # --------- div(f) = trace(Hess g) = 40 * (C*H*W) (constant) ---------
        _, C, H, W = zt.shape
        div_f = 40.0 * float(C * H * W)  # scalar

        # --------- residual per sample: r = div(f) + <f, score> ---------
        dot = (f_field * score).float().sum(dim=(1,2,3))  # (B*num_noise,)
        r = dot + div_f

        # aggregate per original image
        r = r.view(num_images, num_noise)
        stat = r.abs().mean(dim=1)  # (B,)

        ret = []
        for i in range(num_images):
            d = {"criterion": float(stat[i].item())}
            if return_terms:
                d["residual_abs_mean"] = float(stat[i].item())
                d["div_f_const"] = float(div_f)
                d["dot_mean"] = float(r[i].mean().item())
            ret.append(d)

        return ret

    return sdv14_pde_criterion
   
def load_sdv14_criterion_functionalities(device):
    # load main functionalities
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    unet, tokenizer, text_encoder, vae, scheduler = get_sd_v14_model(device)
    image_to_text = pipeline_caption("image-text-to-text", model="llava-hf/llava-1.5-7b-hf", device=device)
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    
    
    return unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text
      
if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_path_real_1 = "example_images/real_car.jpg"
    image_path_real_2 = "example_images/real_dog.png"
    image_path_gen_1 = "example_images/gen_okapi.png"
    image_path_gen_2 = "example_images/gen_frog.png"
    siz = 512
    image_type = 0
    dataset_type = 'sanity'  # 'train' or 'test' usualy
    dataset_name = 'my_collection'  # generative technique or source of real data usually
    num_noise = 8
    time_frac = 0.01
    epsilon_reg = 1e-8
    
    # load functionalities and models
    unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text = load_sdv14_criterion_functionalities(device)
    
    # Example run
    image_paths = [image_path_real_1, image_path_real_2, image_path_gen_1, image_path_gen_2]
    sdv14_based_criterion = factory_sdv14_based_criterion(device, num_noise, epsilon_reg, time_frac, tokenizer, text_encoder, image_to_text, vae, scheduler, unet, clip, processor, cos, siz)
    
    # load images and convert to tensor (still [0,255] range etc..)
    images = load_and_convert_image_batch(image_paths, device)
    
    # run the criterion on the image batch
    res_dict_list = sdv14_based_criterion(images)
    
    # print results
    for idx, (cur_dict, cur_path) in enumerate(zip(res_dict_list, image_paths)):
        print('Image:')
        print(cur_path)
        for key, value in cur_dict.items():
            print(key)
            print(value)
                
    
        
        