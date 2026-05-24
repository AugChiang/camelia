<div align="center">
  <a href="https://github.com/windbow27/camelia">
    <img src="https://raw.githubusercontent.com/windbow27/camelia/refs/heads/main/camelia-ui/public/logo.png" width="125" alt="Camelia Logo">
  </a>
  
  # Camelia
</div>

Camelia is an image decensor tool to remove censorship bars from images (you know what kind of images I am talking about). It supports black bars, white bars, and transparent black censorship types.

## Installation

### Prerequisites

-   Python 3.9
-   Conda (recommended for managing environments)
-   NVIDIA GPU with CUDA support (optional but recommended for faster processing)
-   Node.js 16+ (for Web UI)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/windbow27/camelia
    cd camelia
    ```

2. Create a Conda environment:

    ```bash
    conda create --name camelia_env python=3.9 -y
    conda activate camelia_env
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Verify PyTorch installation:
   Ensure that PyTorch is installed with CUDA support (if applicable):

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

    This should return `True` if CUDA is available.

5. For Web UI, install Node.js dependencies:
    ```bash
    cd camelia-ui
    npm install
    cd ..
    ```

### Models

Download the models here: [models](https://drive.google.com/drive/folders/1AAyv6ms_694VGEtET6TFRLKZ1rx7ue11?usp=sharing)

Put the segmentation models in smp-segmentation/pretrained and inpainting model (the whole folder) in lama-inpainting/pretrained.

## Usage

### CLI Mode

1. Place your input images in the correct `camelia-decensor/input` directory (`input/black_bars`, `input/white_bars`, `input/transparent_black`)

    - Subdirectories can be used

2. Run:

    ```bash
    python main.py --model-type <model-type>
    ```

    Replace `<model-type>` with one of the following options:

    - `black_bars`
    - `white_bars`
    - `transparent_black`

3. The output will be saved under `camelia-decensor/output`.

### Web UI Mode

1. Start the API server, make sure to use the correct environment:

    ```bash
    python api.py
    ```

2. In a separate terminal, start the web UI:

    ```bash
    cd camelia-ui
    npm run dev
    ```

3. Open your browser and navigate to http://localhost:3000

## Acknowledgments

-   [Er0manga](https://github.com/Er0manga/Er0mangaDemo)
-   [smp](https://github.com/qubvel-org/segmentation_models.pytorch)
-   [lama](https://github.com/advimman/lama)
-   [SAM3](https://github.com/facebookresearch/sam3)

## Disclaimer

This project is intended for personal use only. **DO NOT** share the results on public sites. If you choose to do so anyway, please **DO NOT** credit me or Camelia.

Sharing the tool is welcomed. A ☆ would also be greatly appreciated.
