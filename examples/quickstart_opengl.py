###################################################################
########### Use this file for svetoviz_opengl reference ###########
###################################################################

import os

from matplotlib.pyplot import title
from svetoviz_opengl import DiskSceneView, HorizontalLayout

###################################################################
###################### CACHE SETUP ################################
###################################################################
# If you skip this setup, cache data will get stored inside side packages
# make sure you do this BEFORE importing svetoviz
###################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
# Create folder cache_test inside current directory
cache_dir = os.path.join(current_dir, "cache_test")
# Set the environment variable
os.environ["SVETO_VIZ_CACHE_DIR"] = cache_dir

import svetoviz_opengl as sv

# Verify the cache
print(f"Cache dir: {sv.get_cache_dir()}")

# Testing image, replace it with any image URL
IMAGE_URL = "https://img.freepik.com/free-vector/cute-cat-holding-cheek-cartoon-vector-icon-illustration-animal-nature-icon-concept-isolated-premium_138676-4552.jpg"


###################################################################
###################### QUICK OPEN #################################
###################################################################

def array():
    """
    Open single numpy array
    """
    import numpy as np
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    sv.array(data, gap_size=10)


def array_image_channels():
    """
    Open image numpy array
    It renders separate RGBA channels.
    ordering argument is used to reorder two last dimensions to represent height and width
    """
    from PIL import Image
    from io import BytesIO
    import requests
    import numpy as np

    image = Image.open(BytesIO(requests.get(IMAGE_URL).content))
    data = np.array(image)
    sv.array(data, gap_size=10, ordering=(2, 0, 1))


def arrays():
    """
    Open multiple numpy arrays
    """
    import numpy as np
    sv.arrays([
        np.random.randint(-128, 128, (128, 128), dtype=np.int8),
        np.random.randint(-2147483648, 2147483648, (4, 4, 64, 64), dtype=np.int32),
        np.random.uniform(-1e38, 1e38, (32, 32, 32)).astype(np.float32)
    ], gap_size=32)


def tensor():
    """
    Open single torch tensor. Latest pytorch support requires python >= 3.10
    """
    import numpy as np
    import torch

    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    tensor = torch.Tensor(data)
    sv.tensor(tensor, gap_size=10)


def tensors():
    """
    Open multiple torch tensors. Latest pytorch support requires python >= 3.10
    """
    import torch

    sv.tensors(
        [
            torch.randint(0, 2, (2, 72, 72), dtype=torch.bool),
            torch.randint(-128, 128, (2, 60, 60), dtype=torch.int8),
            torch.randint(-32768, 32768, (4, 8, 16, 16), dtype=torch.int16),
            torch.randint(-2147483648, 2147483648, (4, 4, 64, 64), dtype=torch.int32),
            torch.randint(-9223372036854775808, 9223372036854775807, (4, 2, 32, 64), dtype=torch.int64),
            torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8),
            torch.randint(0, 65536, (16, 128, 128), dtype=torch.uint16),
            torch.randint(0, 4294967296, (4, 4, 64, 64), dtype=torch.uint32),
            # torch.frombuffer(np.random.bytes(4 * 2 * 32 * 64 * 8), dtype=torch.uint64).reshape(4, 2, 32, 64),
            (torch.rand(256, 256, dtype=torch.float32) * 65500.0 - 32750.0).to(torch.float16),
            (torch.rand(256, 256, dtype=torch.float32) * 65500.0 - 32750.0).to(torch.bfloat16),
            torch.rand(32, 32, 32, dtype=torch.float32) * 3.4028235e38 * 2 - 3.4028235e38,
            torch.rand(4, 3, 32, 32, dtype=torch.float64) * 1.7976931348623157e308 * 2 - 1.7976931348623157e308,
            torch.complex(
                torch.rand(32, 32, dtype=torch.float32) * 65500.0 - 32750.0,
                torch.rand(32, 32, dtype=torch.float32) * 65500.0 - 32750.0).to(torch.complex32),
            torch.complex(
                torch.rand(32, 32, dtype=torch.float32) * 3.4028235e38 * 2 - 3.4028235e38,
                torch.rand(32, 32, dtype=torch.float32) * 3.4028235e38 * 2 - 3.4028235e38).to(
                torch.complex64),
            torch.complex(
                torch.rand(32, 32, dtype=torch.float64) * 1.7976931348623157e308 * 2 - 1.7976931348623157e308,
                torch.rand(32, 32, dtype=torch.float64) * 1.7976931348623157e308 * 2 - 1.7976931348623157e308)
            .to(torch.complex128)
        ], gap_size=32)


def image():
    """
    Open image. It accepts URL, filepath or PIL.Image object
    """
    sv.image(IMAGE_URL)


def images():
    """
    Open multiple images. Each element can be: URL, filepath or PIL.Image object
    """
    sv.images(imgs=[
        IMAGE_URL,
        IMAGE_URL,
        IMAGE_URL,
    ])


###################################################################
######################### VIEWS API ###############################
###################################################################

def numpy_view():
    """
    Open NumpyView, base view for numpy array
    """
    import numpy as np

    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    view = sv.NumpyView(
        id="array",
        data=data,
        gap_size=10,
        ordering=None)
    sv.view(view)


def numpy_view_image_channels():
    """
    Open NumpyView with separate image channels
    Ordering argument is used to reorder two last dimensions to represent height and width
    """
    from PIL import Image
    from io import BytesIO
    import requests
    import numpy as np

    image = Image.open(BytesIO(requests.get(IMAGE_URL).content))
    data = np.array(image)
    view = sv.NumpyView(
        id="array",
        data=data,
        gap_size=10,
        ordering=(2, 0, 1))
    sv.view(view)


def tensor_view():
    """
    Open TensorView, base view for torch.Tensor
    """
    import numpy as np
    import torch

    data = torch.Tensor(np.random.rand(4, 4, 32, 32).astype(np.float32))
    view = sv.TensorView(
        id="tensor",
        data=data,
        gap_size=10,
        ordering=None)
    sv.view(view)


def disc_image_view():
    """
    Eagerly open image from disc.
    It dumps entire image as memory file into cache directory
    Rendering happens from memory file
    """
    view = sv.DiscImageView(
        id="array",
        image=IMAGE_URL)
    sv.view(view)


def lazy_image_view():
    """
    Lazy open image
    Lazy load the image into RAM when sampled
    """
    view = sv.LazyImageView(
        id="array",
        image=IMAGE_URL)
    sv.view(view)


def vertical_layout():
    """
    Open vertical layout
    Organize views into vertical line
    """
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import requests

    layout = sv.VerticalLayout(
        id="layout",
        spacing="5%",
        views=[
            sv.NumpyView(
                id="array",
                data=np.random.rand(4, 4, 32, 32).astype(np.float32),
                gap_size=10,
                ordering=None),
            sv.NumpyView(
                id="image_array",
                data=np.array(Image.open(BytesIO(requests.get(IMAGE_URL).content))),
                gap_size=10,
                ordering=(2, 0, 1)),
            sv.DiscImageView(
                id="disc_image",
                image=IMAGE_URL),
            sv.LazyImageView(
                id="lazy_image",
                image=IMAGE_URL)
        ],
    )
    sv.view(layout)


def horizontal_layout():
    """
    Open horizontal layout
    Organize views into horizontal line
    """
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import requests
    layout = sv.HorizontalLayout(
        id="layout",
        spacing="5%",
        views=[
            sv.NumpyView(
                id="array",
                data=np.random.rand(4, 4, 32, 32).astype(np.float32),
                gap_size=10,
                ordering=None),
            sv.NumpyView(
                id="image_array",
                data=np.array(Image.open(BytesIO(requests.get(IMAGE_URL).content))),
                gap_size=10,
                ordering=(2, 0, 1)),
            sv.DiscImageView(
                id="disc_image",
                image=IMAGE_URL),
            sv.LazyImageView(
                id="lazy_image",
                image=IMAGE_URL)
        ],
    )
    sv.view(layout)


def grid_layout():
    """
    Open grid layout
    Organize views into grid
    """
    import numpy as np
    layout = sv.GridLayout(
        id="layout",
        views=[
            sv.NumpyView(
                id="array1",
                data=np.random.rand(6, 2, 128, 32).astype(np.float32),
                gap_size=10,
                ordering=None),
            sv.DiscImageView(
                id="disc_image",
                image=IMAGE_URL),
            sv.NumpyView(
                id="array2",
                data=np.random.rand(4, 32, 64).astype(np.float32),
                gap_size=10,
                ordering=None),
            sv.NumpyView(
                id="array3",
                data=np.random.rand(4, 12, 2, 32).astype(np.float32),
                gap_size=10,
                ordering=None),
            sv.NumpyView(
                id="array4",
                data=np.random.rand(2, 2, 12, 2, 24, 24).astype(np.float32),
                gap_size=10,
                ordering=None),
            sv.LazyImageView(
                id="lazy_image",
                image=IMAGE_URL)
        ],
    )
    sv.view(layout)


def grid_layout_styled():
    """
    Open custom layout
    Layouts can be nested to build complex structures
    Control spacing, padding and colors
    Add title argument to render group name
    """
    import numpy as np
    view = sv.GridLayout(
        id="layout",
        title="Grid Layout",
        padding="2%",
        title_color=(155, 155, 155),
        background_color=(0, 0, 0, 255),
        spacing=40,
        views=[
            sv.VerticalLayout(
                id="vertical1",
                title="np.float32",
                padding="5%",
                spacing="5%",
                background_color=(50, 50, 50, 255),
                views=[
                    sv.NumpyView(
                        id="float32",
                        data=np.random.rand(
                            6, 24,
                            128, 32).astype(np.float32),
                        gap_size=10,
                        ordering=None)
                ]
            ),
            sv.VerticalLayout(
                id="vertical2",
                title="Images",
                padding="5%",
                spacing="5%",
                background_color=(50, 50, 50, 255),
                views=[
                    sv.DiscImageView(
                        id="disc_image",
                        image=IMAGE_URL),
                    sv.LazyImageView(
                        id="lazy_image",
                        image=IMAGE_URL)
                ]
            ),

            sv.VerticalLayout(
                id="float64_container",
                title="np.float64",
                padding="5%",
                background_color=(50, 50, 50, 255),
                views=[
                    sv.NumpyView(
                        id="float64",
                        data=(np.random.uniform(
                            -1, 1,
                            (2, 32, 72, 24))
                              * 1.7976931348623157e+308).astype(np.float64),
                        gap_size=10,
                        ordering=None)
                ]
            ),
            sv.HorizontalLayout(
                id="vertical3",
                title="Integer",
                padding="5%",
                spacing="5%",
                gravity="center",
                background_color=(50, 50, 50, 255),
                views=[
                    sv.VerticalLayout(
                        id="int8_container",
                        title="np.int8",
                        padding="5%",
                        background_color=(50, 50, 50, 255),
                        views=[
                            sv.NumpyView(
                                id="int8",
                                data=np.random.randint(
                                    -128, 128,
                                    (2, 2, 128, 128), dtype=np.int8),
                                gap_size=10,
                                ordering=None)
                        ]
                    ),
                    sv.VerticalLayout(
                        id="int32_container",
                        title="np.int32",
                        padding="5%",
                        background_color=(50, 50, 50, 255),
                        views=[
                            sv.NumpyView(
                                id="int32",
                                data=np.random.randint(
                                    -2147483648, 2147483648,
                                    (4, 4, 64, 64), dtype=np.int32),
                                gap_size=10,
                                ordering=None)
                        ]
                    ),
                ]
            ),

        ],
    )
    sv.view(view)


def directory_grid_layout():
    """
    Open directory of images organized into a grid
    Use limit argument to open only N first images
    Use lazy argument to use DiscImageView or LazyImageView
    """

    layout = sv.DirectoryGridView(
        id="directory_grid",
        directory="/Users/piotrgryko/Desktop",
        limit=100,
        lazy=False
    )

    # For small numbers of images you can open directly using sv.view(layout)
    # If using lazy=True you are limited by:
    # - Available RAM
    # - You may see empty scene at launch until data loads
    # If using lazy=False you are limited by:
    # - Available disc space for DiscImageView memory files
    # - Max opened file descriptors count

    # sv.view() may start failing around 500-1000 data points.
    # Launch loading times may get unacceptable with lazy=True,
    # or you may hit max opened descriptors error with lazy=False
    sv.view(layout)
    return
    # For large number of images save your scene offline
    # After saving the scene can be opened without original images with minimal loading times
    # Use compression argument to control:
    # - Used disc space
    # - Scene quality
    # - Build duration
    # Use image_backend argument to process images using
    # - pillow
    # - opencv
    # Select one that works better
    # Use max_memory_gb argument to set loose max memory cap
    # max_memory_gb is used only for lazy images pool
    output_directory = f"/{sv.get_cache_dir()}/directory_layouts"
    sv.save_to_disc(view=layout,
                    threaded=True,
                    directory=output_directory,
                    compression="png_0",
                    image_backend="pillow")

    # Open directly from output_directory
    sv.load_from_disc(directory=output_directory)


def directory_spiral_layout():
    """
    Open directory of images organized into a spiral
    Use limit argument to open only N first images
    Use lazy argument to use DiscImageView or LazyImageView
    """

    #### Spiral is slower to measure. For large scenes prefer DirectoryGridLayout
    layout = sv.DirectorySpiralView(
        id="directory_spiral",
        directory="/Users/piotrgryko/Desktop",
        limit=100,
        lazy=False
    )

    # For small numbers of images you can open directly using sv.view(layout)
    # If using lazy=True you are limited by:
    # - Available RAM
    # - You may see empty scene at launch until data loads
    # If using lazy=False you are limited by:
    # - Available disc space for DiscImageView memory files
    # - Max opened file descriptors count
    # sv.view() may start failing around 500-1000 data points.
    # Launch loading times may get unacceptable with lazy=True,
    # or you may hit max opened descriptors error with lazy=False
    sv.view(layout)
    return
    # For large number of images save your scene offline
    # After saving the scene can be opened without original images
    # Use compression argument to control:
    # - Used disc space
    # - Scene quality
    # - Build duration
    # Use image_backend argument to process images using
    # - pillow
    # - opencv
    # Select one that works better
    # Use max_memory_gb argument to set loose max memory cap
    # max_memory_gb is used only for lazy images pool
    output_directory = f"/{sv.get_cache_dir()}/directory_layouts"
    sv.save_to_disc(view=layout,
                    threaded=True,
                    directory=output_directory,
                    compression="png_0",
                    image_backend="pillow")

    # Open directly from output_directory
    sv.load_from_disc(directory=output_directory)


def dataset_subset():
    """
    Open HF dataset subsets
    Use DatasetView to render dataset grid view
    DatasetView can be placed into layouts for grouping

    Open using sv.view(layout) for small amount of data points
    - You are limited by available RAM
    - Gets slow around 500-1000 images

    For large scenes use sv.save_to_disc and sv.load_from_disc
    - You are limited by available disc space

    """
    from datasets import load_dataset

    hf_name = "Dewa/Dog_Emotion_Dataset_v2"
    dataset = load_dataset(hf_name, split="train")
    subset = dataset.select(range(100))
    subset2 = dataset.select(range(100, 200))

    view1 = sv.DatasetView(
        id="subset1",
        dataset=subset,
        image_columns=["image"],
        cell_width=400,
        cell_height=400,
    )
    view2 = sv.DatasetView(
        id="subset2",
        dataset=subset2,
        image_columns=["image"],
        cell_width=400,
        cell_height=400,
    )

    layout = sv.HorizontalLayout(
        id="vertical1",
        title="Dog_Emotion_Dataset_v2",
        padding="5%",
        spacing="5%",
        background_color=(50, 50, 50, 255),
        views=[
            sv.VerticalLayout(
                id="subset1_layout",
                title="Subset1",
                padding="5%",
                views=[view1],
            ),
            sv.VerticalLayout(
                id="subset2_layout",
                title="Subset2",
                padding="5%",
                views=[view2],
            )
        ])
    sv.view(layout)


def dataset_full_small():
    """
    Open full HF dataset
    Use DatasetView to render dataset grid view

    For large scenes use sv.save_to_disc and sv.load_from_disc
    You are limited by available disc space

    """
    from datasets import load_dataset

    hf_name = "Dewa/Dog_Emotion_Dataset_v2"
    dataset = load_dataset(hf_name, split="train")
    layout = sv.DatasetView(
        id="dog_emotion",
        dataset=dataset,
        image_columns=["image"],
        cell_width=400,
        cell_height=400,
    )

    # Set your output directory
    dir = f"/Volumes/Untitled/{hf_name}"
    sv.save_to_disc(view=layout,
                    threaded=True,
                    directory=dir,
                    compression="jpeg_0",
                    image_backend="opencv")
    sv.load_from_disc(directory=dir)


def dataset_full_medium():
    """
    Open HF dataset subset
    Use DatasetView to render dataset grid view

    For large scenes use sv.save_to_disc and sv.load_from_disc
    You are limited by available disc space

    """
    from datasets import load_dataset

    hf_name = "eltorio/ROCOv2-radiology"
    dataset = load_dataset(hf_name, split="train")
    layout = sv.DatasetView(
        id=hf_name,
        dataset=dataset,
        image_columns=["image"],
        info_columns=("image_id", "caption", "cui"),
        cell_width=400,
        cell_height=400
    )

    # Set your output directory
    dir = f"/Volumes/Untitled/{hf_name}"
    sv.save_to_disc(view=layout,
                    threaded=True,
                    directory=dir,
                    compression="jpeg_0",
                    image_backend="opencv")
    sv.load_from_disc(directory=dir)

def disk_scene_view():
    """
    Use DiskSceneView to render saved scene as a Widget

    """
    # Saved Dog_Emotion_Dataset_v2 dataset
    dog_emotions = DiskSceneView(
        id="dog_emotion",
        directory=f"/Volumes/Untitled/Dewa/Dog_Emotion_Dataset_v2",
    )
    # Saved ROCOv2-radiology  dataset
    radiology = DiskSceneView(
        id="radiology",
        directory=f"/Volumes/Untitled/eltorio/ROCOv2-radiology",
    )
    layout = HorizontalLayout(
        id="container",
        title="Datasets",
        spacing="5%",
        padding="5%",
        background_color=(50, 50, 50, 255),
        views=[dog_emotions, radiology]
    )
    sv.view(layout)


###################################################################
########################  PYTORCH DEBUGGER ########################
###################################################################

# Collection of all supported pytorch modules
# Pass any string into terminal view to process the modules
# Use timeline view to go through activations
def modules_demo():
    """
    Open collection of all supported modules
    These examples are fixed, if you want to use real data, activate the product
    """
    sv.demo_modules()


# Render any pytorch module with sv.pytorch()
# It uses module.named_children() for layout tree
# It uses module.weight and module.bias for data points
# Module activations are collected
# See input and output for each activation
# Select data point to see the relations

# To open the module

# Open any pytorch module
# my_module = nn.Module(args)
# my_input_data = ...
#

# Define GUI terminal view callback
# Module interactions inside terminal_callback are collected automatically

# def terminal_callback(buffer, message, images, files):
#     output = my_module(my_input_data)
#     buffer.send_system_message(f"Input: {list(input_tensor.shape)}")
#     buffer.send_system_message(f"Output: {list(output.shape)}")
# Use GUI timeline view to see the result
# Select the output value to see details

# Open the module
# sv.pytorch(module=my_module, terminal_callback=terminal_callback)


def pytorch_maxpool1d():
    # 1. Create a synthetic 1D signal (e.g., a noisy sine wave)
    import torch
    from torch import nn

    t = torch.linspace(0, 10, 100)
    signal = torch.sin(t) + torch.randn(100) * 0.2
    # Shape: (Batch=1, Channels=1, Length=100)
    input_tensor = signal.unsqueeze(0).unsqueeze(0)

    # 2. Define MaxPool1d
    # Kernel size 4 reduces the resolution by taking the max of every 4 points
    pool1d = nn.MaxPool1d(kernel_size=4, stride=2)

    def terminal_callback(buffer, message, images, files):
        output = pool1d(input_tensor)
        buffer.send_system_message(f"Input Signal: {list(input_tensor.shape)}")
        buffer.send_system_message(f"Downsampled Signal: {list(output.shape)}")

    sv.pytorch(module=pool1d, terminal_callback=terminal_callback)


def pytorch_conv2d():
    from PIL import Image
    import numpy as np
    from torch import nn

    # 1. Prepare 2D image data
    # Load image and convert to (Batch, Channels, H, W) tensor
    img = Image.open("sample_image.jpg").convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    # Rearrange from (H, W, C) to (C, H, W) and add batch dim
    import torch
    input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # 2. Define the 2D Convolutional module
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    def terminal_callback(buffer, message, images, files):
        # 3. Process the image through the module
        output = conv2d(input_tensor)

        # Log the feature map shape to the TensorViz terminal
        buffer.send_system_message(f"Processed image. Shape: {list(output.shape)}")

    # 4. Start the interactive session
    sv.pytorch(module=conv2d, terminal_callback=terminal_callback)


def gpt2():
    from transformers import GPT2Tokenizer
    from transformers import GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', device_map="cpu")
    model = GPT2LMHeadModel.from_pretrained('gpt2', device_map="cpu")

    def terminal_callback(buffer, message, images, files):
        print("terminal called")
        text = message  # "Replace me by any text you'd like."
        buffer.send_user_message(text)
        inputs = tokenizer(text, return_tensors="pt")

        # Generate continuation
        outputs = model.generate(**inputs, max_new_tokens=50)

        # Decode to string
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded)
        buffer.send_system_message(decoded)

    sv.model(
        id="gpt2",
        model=model,
        tokenizer=tokenizer,
        terminal_callback=terminal_callback
    )


def yolos_object_detection():
    from transformers import YolosForObjectDetection
    from transformers import YolosImageProcessor
    from PIL import Image
    from io import BytesIO
    import requests

    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny', device_map="cpu")
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny", device_map="cpu")

    def terminal_callback(buffer, message, images, files):
        for image in images:
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            buffer.send_system_message("processed")

    sv.model(
        id="yolos",
        model=model,
        image_processor=image_processor,
        terminal_callback=terminal_callback
    )


def stable_diffusion():
    from diffusers import DiffusionPipeline
    import torch

    pipe = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()

    def terminal_callback(buffer, message, images, files):
        # buffer.send_system_message(message)
        # image = pipe(message).images[0]
        # image.show("output.png")
        buffer.send_system_message("This will fill up your memory! Use WebGPU version to use breakpoints")

    sv.pipeline(id="stable_diffusion",
                pipe=pipe,
                terminal_callback=terminal_callback)


if __name__ == "__main__":
    array()
    # array_image_channels()
    # arrays()

    # tensor()
    # tensors()

    # image()
    # images()

    # numpy_view()
    # numpy_view_image_channels()

    # tensor_view()

    # disc_image_view()
    # lazy_image_view()

    # vertical_layout()
    # horizontal_layout()
    # grid_layout()

    # grid_layout_styled()

    # directory_grid_layout()
    # directory_spiral_layout()

    # dataset_subset()
    # dataset_full_small()
    # dataset_full_medium()
    # disk_scene_view()

    # modules_demo()

    # pytorch_maxpool1d()
    # pytorch_conv2d()

    # gpt2()
    # yolos_object_detection()
    # stable_diffusion()
