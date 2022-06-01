### <em>IU000133 : Artificial Intelligence for Media</em>
# Speech2expression:Change the Expression Behind a Voice
![](https://github.com/Leonie622/Speech2expression/blob/main/IALS-main/image_output/happy/001.wav377e.jpg?raw=true)
## Project Rationale
The human voice contains more information than we can imagine, your age, height, weight, spirit, mood, even emotion, temperament, drive, creativity, and so on. And in the virtual world nowadays, the voice gives an additional personality label to a person, giving another imaginary form of expression to others. In this project plan, I would like to go through the classification of emotions extracted from speech and try to make the face of a face image change accordingly due to the speech under different emotions, which to explore the unknown effects of diverse voices on the generation of facial expressions.

## Project Description
This project refers to multiple open source projects such as Speech-Emotion-Analyzer, interfacegan,IALS,Stylegan,pix2pixhd. Through my own understanding, the data set ( Multi-view Emotional Audio-visual Dataset (MEAD)) is selected, subjected to repeated training and testing, and I carried out a detailed explanation and description.

## Dataset
The data source comes from Multi-view Emotional Audio-visual Dataset (MEAD), data link: https://wywu.github.io/projects/MEAD/MEAD.html
| Data        | Download Link |
| ----------- | -------------|
| MEAD(Initial Data)| [Google Drive](https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j)|

The original data is Audio-visual Dataset downloaded directly from the Internet. for this project, I have selected only some of the datasets with the five more variable mood categories of "angry", "neutral", "fear", "happy" and "sad" as the generation and testing datasets. "happy", and "sad", which are five highly variable mood categories, were used as the generation and test dataset.

## Requirements 
* `Python 3.7 and the basic Anaconda3 environment. `
* `PyTorch 1.x with GPU support.`
* `The tqdm library.`

## Document Instruction
This project consists of two aspects: `Voice recognition`, `Facial expression generation`
### Video recognition
In this module use Librosa's MFCC method to extract high-dimensional MFCC information from voice [Speech-Emotion-Analyzer](https://github.com/DimensionNXG/Speech-Emotion-Analyzer) and use a Linear layer to realize audio classification features. 
- Recognising audio emotions: `Speech2Expression/speech_emotion/inference.py`.define predict_emotion module to assign five emotion labels:`angry`,`happy`,`netural`,`fear`,`sad`

### Facial expression generation
In this module I have tried two approaches to facial emotion modification： 

- pix2pixHD：`"pix2pix_approach" folder`
  - Extract video frames and stored in folder corresponding to emotions: `pix2pix_approach/propressed_dataset.py`
  - Set up five corresponding signs based on the five emotions: ``pix2pix_approach/draw_sign_image.py`
  - Draw emotion signs on corresponding emotion images by CV: `pix2pix_approach/utils.py`,`pix2pix_approach/infer.py`
  - Matched signed images with the original images to form a pix2pix dataset
  - Data augmentation of datasets:`pix2pix_approach/pix2pixhd/draw_sign_image.py`
  - Implementing pix2pix model to complete face generation:though `pix2pix_approach/pix2pixhd/infer.py`
   
- IALS：`"IALS-main" folder" folder`
  - Processed the input face image and aligned it to FFHQ face format: `Xiaolin Deng-AI-Speech2expression/static_filesdlib_align_face.py`
  - convert video format and generate emotion labels: `extract_audio.py`convert the .mp4 video in the original dataset to .wav format.
  - Imported the values of the parameters corresponding to the generated sentiment labels into the IALS framework and edit a real face image via instance-aware direction:`Xiaolin Deng-AI-Speech2expression/infer.py`
   ```
   # matched smile and young condition attributes for primal attribute editing
   cmd1 = "python gan_inversion.py --n_iters 100  --img_path {}".format(img_path)
   cmd2 = "python condition_manipulation.py --seed 0 --step -0.1 --n_steps 10 --dataset ffhq --base interfacegan\--attr1 smiling --attr2 young --lambda1 {} --lambda2 0.3 --real_image 1 --latent_code_path rec.npy --save_path {}".format(e_lamdbda[ret_emotion], out_img_path)
   ```
  - Repeatedly adjust the weight parameters to fit and generate facial expression images.   
  ```
  #Set emotion weight parameters
   e_lamdbda = {
      "angry":-100.,
      "netural":1.2,
      "fear":-1.,
      "sad":0.2,
      "happy":100.5
   }
  ```
- Generates facial images that change expressions according to the voice:it is stored in the `"IALS-main/image_output"` path and categorised by the five emotions.In this project, we use same voice content, with highest emotional level, five emotional features, and different people's voices as input dataset.

