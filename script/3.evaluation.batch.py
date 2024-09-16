#!/usr/bin/env python
# coding: utf-8

# # Batch Evaluation of InternVL
#
# Author: Rongxin Ouyang
# Email: rongxin@u.nus.edu
# Date: 2024-08-29

# ## Imports

# In[37]:


import math
import os
import re
import warnings
from dataclasses import dataclass

import pandas as pd
import torch
import torchvision.transforms as T
from pandarallel import pandarallel
from PIL import Image
from rich import print as pp

# Accuracy, f1 score, precision, recall, and AUROC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

tqdm.pandas()

pandarallel.initialize(progress_bar=True, nb_workers=10)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")


# ## Configurations

# ### Variables

# In[38]:


sample_size = 0
img_path = "../dataset/raw/hateful_memes_expanded/img/"
with_human_labels = True
prompts_to_use = ["simple", "simple_scale", "category", "category_scale"]


# ### Functions

# In[39]:


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# batch inference, single image per sample (单图批处理)
# Author: Rongxin
def load_images(image_paths, max_num=12):
    pixel_values_list = []
    num_patches_list = []
    for image_path in image_paths:
        pixel_values = load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


# run_single_eval()
# run_batch_eval(prompt=getattr(prompt, "simple") + prompt.constrain)


# ### Prompts

# In[40]:


@dataclass
class prompt:
    simple_scale: str
    category_scale: str
    simple: str = "\nYour task is to analyze this given image and its caption to identify if there's any forms of hateful content."
    category: str = "\nTry to focus on the presence of any element that relates to any of the following:"
    definition: str = "\n1. Sexual aggression:\na. Homophobia and Transphobia: This category encompasses hate speech targeting LGBTQ+ individuals, including slurs, derogatory comments, and rhetoric that seeks to undermine or dehumanize people based on their sexual orientation or gender identity.\nb. Misogyny and Sexism: This category includes hate speech directed at women or based on gender. It covers derogatory language, stereotypes, and rhetoric that perpetuate gender inequality, objectification, and violence against women.\n2. Hate based on ideology:\na. Political Hate Speech: This category includes hate speech that is politically motivated, often targeting individuals or groups based on their political beliefs or affiliations. It may include inflammatory language, threats, and rhetoric designed to polarize or incite violence within political contexts.\n3. Racism and xenophobia:\na. COVID-19 and Xenophobia: This category includes hate speech that arose during the COVID-19 pandemic, often targeting specific ethnic groups or nationalities. It includes xenophobic language blaming certain groups for the spread of the virus, as well as fear-mongering and scapegoating related to the pandemic.\nb. Racism Against Black People: This category focuses on hate speech directed at Black individuals or communities. It includes racial slurs, stereotypes, dehumanization, and other forms of derogatory language that perpetuate racial discrimination and inequality.\nc. Racist Hate Against Other Ethnic Groups: This category includes hate speech directed at various ethnic groups other than Black individuals. It covers a range of racial slurs, xenophobic language, dehumanization, and derogatory remarks targeting specific ethnicities or nationalities.\nd. White Supremacy: This category includes hate speech promoting white supremacist ideology, often intertwined with Christian extremist views. It includes rhetoric that advocates for racial superiority, anti-immigrant sentiments, and the dehumanization of non-white groups, sometimes using religious justifications for these beliefs.\n4. Bigotry:\na. Anti-Muslim and Islamophobic Hate: This category comprises hate speech aimed at Muslims and Islam. It includes language that promotes fear, hatred, dehumanization, or prejudice against Muslim individuals or communities, often using religious and cultural references to incite hostility.\nb. Anti-Semitic Hate: This category focuses on hate speech directed at Jewish people and Judaism. It includes references to historical anti-Semitic tropes, conspiracy theories, and other forms of rhetoric that seek to dehumanize or discriminate against Jewish individuals and communities.\n5. Miscellaneous Hate Speech: This category captures various forms of hate speech that do not fit neatly into the other categories. It includes a wide range of derogatory language and expressions that target individuals or groups based on different aspects of identity or personal characteristics. This category includes hate speech that targets individuals based on their physical or mental disabilities. It often includes derogatory language that mocks or devalues people with disabilities, promoting harmful stereotypes and exclusion."

    scale: str = "\nYou will be provided with the image, based on this label and the content of the image, you will assign a hatefulness score on a scale from 0 to 9, where:\n- 0: The image is not hateful at all.\n- 9: The image is extremely hateful."
    constrain: str = "\nIf it's hateful, return the `TRUE` | `FALSE`."
    constrain_int: str = "\nReturn the score as an integer in range 0 to 9."


# ADD CONSTRAINTS
prompt.scale = prompt.scale + prompt.constrain_int
prompt.simple_scale = prompt.simple + prompt.scale
prompt.category = prompt.simple + prompt.category + prompt.definition
prompt.category_scale = prompt.category + prompt.scale
prompt.category = prompt.category + prompt.constrain
prompt.simple = prompt.simple + prompt.constrain
pp(f"[bold]Simple:[/bold]{prompt.simple}")
pp(f"[bold]Simple + Scale:[/bold]{prompt.simple_scale}")
pp(f"[bold]Category:[/bold]{prompt.category}")
pp(f"[bold]Category + Scale :[/bold]{prompt.category_scale}")


# ## Modelling

# ### Functions

# In[41]:


# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

use_multiple_gpus = True
# use_multiple_gpus = False

model_size = "8B"  # | "2B"
# model_size = "2B"  # | "8B"

path = f"OpenGVLab/InternVL2-{model_size}"
if use_multiple_gpus:
    device_map = split_model(f"InternVL2-{model_size}")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
else:
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


def run_batch_eval(
    images: list = ["0Jzts4J.png", "V2wUrRj.png"],
    prompt: str = "Is this image hateful?",
    image_folder_path: str = img_path,
    model=model,
):
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    image_paths = [f"{image_folder_path}{image}" for image in images]

    def load_images(image_paths, max_num=12):
        pixel_values_list = []
        num_patches_list = []
        for image_path in image_paths:
            pixel_values = (
                load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
            )
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list

    pixel_values, num_patches_list = load_images(image_paths)

    prompt = "<image>\n" + prompt
    questions = [prompt] * len(num_patches_list)
    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config,
    )
    return responses


def run_single_eval(
    image: str = "0Jzts4J.png",
    prompt: str = "Is this image hateful?",
    context: str = "",
    image_folder_path: str = img_path,
    with_human_labels: bool = True,
):
    pixel_values = (
        load_image(f"{image_folder_path}{image}", max_num=12).to(torch.bfloat16).cuda()
    )
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    if context != "" and with_human_labels:
        prompt = "<image>\nThis image is about " + context + ".\n" + prompt
    else:
        prompt = "<image>\n" + prompt
    question = prompt
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


# ### Data

# Load the evaluation table:

# In[47]:


df = pd.read_json(
    "../dataset/raw/hateful_memes_expanded/train.jsonl",
    lines=True,
)[["img", "text", "label"]]
df["type"] = "train"

df_tmp = pd.read_json(
    "../dataset/raw/hateful_memes_expanded/dev_seen.jsonl",
    lines=True,
)[["img", "text", "label"]]
df_tmp["type"] = "dev"

df = pd.concat(
    [
        df,
        df_tmp,
    ]
)
df_tmp = pd.read_json(
    "../dataset/raw/hateful_memes_expanded/dev_unseen.jsonl",
    lines=True,
)[["img", "text", "label"]]
df_tmp["type"] = "dev"

df = pd.concat(
    [
        df,
        df_tmp,
    ]
)


df_tmp = pd.read_json(
    "../dataset/raw/hateful_memes_expanded/test_seen.jsonl",
    lines=True,
)[["img", "text", "label"]]
df_tmp["type"] = "test"

df = pd.concat(
    [
        df,
        df_tmp,
    ]
)
df_tmp = pd.read_json(
    "../dataset/raw/hateful_memes_expanded/test_unseen.jsonl",
    lines=True,
)[["img", "text", "label"]]
df_tmp["type"] = "test"

df = pd.concat(
    [
        df,
        df_tmp,
    ]
)


df["img"] = df["img"].apply(lambda x: x.split("/")[-1])
pp(df[df["type"] == "test"].value_counts("label"))
df.head()


# Filter not existed:

# In[48]:


df["existed"] = df["img"].parallel_apply(
    lambda x: 1 if os.path.exists(f"{img_path}{x}") else 0
)
df = df[df["existed"] == 1]
pp(df.shape)


# In[49]:


df_train = df[df["type"] == "train"]
df_dev = df[df["type"] == "dev"]
df_train_dev = pd.concat([df_train, df_dev])
df_test = df[df["type"] == "test"]
pp(df_train.shape, df_dev.shape, df_test.shape)
df_train_dev.head()


# ## Evaluation

# ### Sampling

# In[50]:


if sample_size != 0:
    df_eval = df_test.sample(sample_size, random_state=42)
    pp(f"Attention: Sample size: {sample_size}")
else:
    df_eval = df_test.copy()
df_eval.head()


# ### Run

# In[51]:


df_eval["tuple"] = df_eval.apply(lambda x: (x["img"], x["text"]), axis=1)
for prompt_name in prompts_to_use:
    pp(f"[bold blue]Evaluating {prompt_name} on N={len(df_eval)}...[/bold blue]")

    df_eval[prompt_name + "_r"] = df_eval["tuple"].progress_apply(
        lambda x: run_single_eval(x[0], getattr(prompt, prompt_name), context=x[1])
    )
df_eval.head()


# ## Output

# ### Prepare

# In[ ]:


current_time = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
result_path = "../result/internvl.aaai/"
if sample_size != 0:
    file_name = f"{current_time}.sample.{sample_size}"
else:
    file_name = f"{current_time}.full.{len(df_eval)}"

# Make result directory if not exists
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(result_path + "evaluation/"):
    os.makedirs(result_path)
if not os.path.exists(result_path + "result/"):
    os.makedirs(result_path)
if not os.path.exists(result_path + "prompt/"):
    os.makedirs(result_path)


def convert_to_label(x, scale=False):
    if scale:
        try:
            if int(x) <= 4:
                return 0
            else:
                return 1
        except Exception as e:
            pp(e)
            try:
                pattern = r"\b.*?(\d+).*?\b"
                matches = re.findall(pattern, x)
                x = int(matches[0])
                if x <= 4:
                    return 0
                else:
                    return 1
            except Exception as e:
                pp(e)
                if "I can not" in x:
                    return 1
                elif "I can't" in x:
                    return 1
                elif "TRUE" in x:
                    return 1
                elif "FALSE" in x:
                    return 0
                elif "`TRUE`" in x:
                    return 1
                elif "`FALSE`" in x:
                    return 0
                elif "is hateful" in x:
                    return 1
                elif "is not hateful" in x:
                    return 0
                elif "not hateful" in x:
                    return 0
                else:
                    return -1
    else:
        if "I can not" in x:
            return 1
        elif "I can't" in x:
            return 1
        elif "TRUE" in x:
            return 1
        elif "FALSE" in x:
            return 0
        elif "`TRUE`" in x:
            return 1
        elif "`FALSE`" in x:
            return 0
        elif "is hateful" in x:
            return 1
        elif "is not hateful" in x:
            return 0
        else:
            return -1


df_eval["label"] = df_eval["label"].astype(int)

for prompt_name in prompts_to_use:
    col_name = prompt_name + "_r"
    if col_name in df_eval.columns:
        if "_scale" in prompt_name:
            df_eval[col_name + "_r_b"] = df_eval[prompt_name + "_r"].parallel_apply(
                lambda x: convert_to_label(x, scale=True)
            )
        else:
            df_eval[col_name + "_r_b"] = df_eval[prompt_name + "_r"].parallel_apply(
                lambda x: convert_to_label(x, scale=False)
            )
        # Check if there's any -1
        if -1 in df_eval[col_name + "_r_b"].unique():
            pp(f"[bold red]Warning: {prompt_name} has -1[/bold red]")
    else:
        pass

df_eval.to_csv(f"{result_path}evaluation/{file_name}.csv", index=False)
pp(f"[bold]Saved to {result_path}evaluation/{file_name}.csv[/bold]")


# ### Construct a table

# In[ ]:


prompts_result = {}
for prompt_name in prompts_to_use:
    col_name = prompt_name + "_r"
    if col_name in df_eval.columns:
        try:
            accu = f"{accuracy_score(df_eval['label'], df_eval[col_name + '_r_b']) * 100:.3f}"
        except Exception as e:
            accu = 1
            pp(
                f"[bold red]Accu failed on {prompt_name} for {e};\n use 100% instead. [/bold red]"
            )
        try:
            f1 = f"{f1_score(df_eval['label'], df_eval[col_name + '_r_b']) * 100:.3f}"
        except Exception as e:
            f1 = 1
            pp(
                f"[bold red]F1 failed on {prompt_name} for {e};\n use 100% instead. [/bold red]"
            )
        try:
            precision = f"{precision_score(df_eval['label'], df_eval[col_name + '_r_b']) * 100:.3f}"
        except Exception as e:
            precision = 1
            pp(
                f"[bold red]Precision failed on {prompt_name} for {e};\n use 100% instead. [/bold red]"
            )

        try:
            recall = f"{recall_score(df_eval['label'], df_eval[col_name + '_r_b']) * 100:.3f}"
        except Exception as e:
            recall = 1
            pp(
                f"[bold red]RECALL failed on {prompt_name} for {e};\n use 100% instead. [/bold red]"
            )
        try:
            auroc = f"{roc_auc_score(df_eval['label'], df_eval[col_name + '_r_b']) * 100:.3f}"
        except Exception as e:
            auroc = 1
            pp(
                f"[bold red]AUROC failed on {prompt_name} for {e};\n use 100% instead. [/bold red]"
            )
        prompts_result[prompt_name] = {
            "prompt": prompt_name,
            "accuracy": accu,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
        }
        pp(
            f"[bold]Prompt: {prompt_name}[/bold]\n[Accuracy]: {accu}%  [F1-score]: {f1}%  [AUROC]: {auroc}%"
        )
    else:
        pass
results = pd.DataFrame(prompts_result).T
results.to_csv(f"{result_path}result/{file_name}.csv", index=True)
pp(f"[bold green]Results saved to: {file_name}.csv[/bold green]")
results.head()


# ### Save

# Save prompts
with open(f"{result_path}/prompt/{file_name}.prompts.txt", "w") as f:
    for prompt_name in prompts_to_use:
        f.write(f"{getattr(prompt, prompt_name)}\n")
    f.close()

pp(f"[bold green]Prompts saved to: {file_name}.prompts.txt[/bold green]")
