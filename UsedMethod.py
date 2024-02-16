#Instead of using pytorch to make the cnn, i used fast ai, which provided a much better validation loss, and a more accurate cnn, and is easier to use make responses with.
#My attempt at pytorch is in the file called "PytorchMethod.ipynb"
from fastai.vision.all import*
import torch
print(torch.cuda.is_available())
from fastai.vision.all import DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, get_image_files
from fastai.vision.widgets import ImageClassifierCleaner
import zipfile
from pathlib import Path
from PIL import Image

def PrepareDataset():
    #download zip file from https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification?rvi=1
    #move to root directory

    #unzips in root directory
    with zipfile.ZipFile('cards-image-datasetclassification.zip', 'r') as zip_ref:
        zip_ref.extractall('path_to_extract_to')
    #variables 
path = Path('./path_to_extract_to') # path to extracted zip file
test_path = path / 'test' / 'test' # path to test folder that will be removed
valid_path = path / 'valid' # path to valid folder that will be removed
files_to_remove = ['14card types-14-(200 X 200)-94.61.h5', '53cards-53-(200 X 200)-100.00.h5'] # files to remove
#remove test and valid folders if they exist
if test_path.exists() and test_path.is_dir(): # if test_path exists and is a directory
    shutil.rmtree(test_path) # remove test_path
if valid_path.exists() and valid_path.is_dir(): # if valid_path exists and is a directory
    shutil.rmtree(valid_path) # remove valid_path
#remove files that are needed
for file_name in files_to_remove: # for each file in files_to_remove
    file_path = path / file_name # get path to file
    if file_path.exists() and file_path.is_file(): # if file exists and is a file
        os.remove(file_path) # remove file
#resize images to 224x224 & ensure they are jpg
path = Path('./path_to_extract_to/train')
for folder in path.iterdir(): # for each folder in train
    if folder.is_dir(): # if folder is a directory
        for image in folder.iterdir(): # for each image in folder
            imagePath = Path(f'./path_to_extract_to/train/{folder}/{image}') # get path to image {ace of clubs}/{ace of clubs/1.jpg}
            if image.is_file() and image.suffix.lower() in ['.jpg', '.jpeg']:  # Check if file is a JPG image # check if image is a file and is a jpg
                os.remove(imagePath) # remove image
                print(f'removed {imagePath}') # print removed image
            else: # if image is a file and is a jpg
                with Image.open(imagePath) as img: # open image
                    if img.size != (224,224):# if image is not 224x224
                        img = img.resize((224,224))# resize image
                        img.save(imagePath)# save image
                        print(f'this image had to be resized {imagePath}')# print resized image
#main

if __name__ == '__main__':
    PrepareDataset()
    path = Path('./path_to_extract_to/train')
    cards = DataBlock(
        blocks= (ImageBlock, CategoryBlock),
        get_items = get_image_files,
        splitter=RandomSplitter(valid_pct=.2,seed=42),
        get_y=parent_label,
        item_tfms=Resize(224)
    ).dataloaders(path,bs=40)


    max_images = 120
    n_cols = 10
    n_rows = max_images // n_cols + (max_images % n_cols != 0)
    cards.show_batch(max_n=max_images, nrows=n_rows, ncols=n_cols)


    learn = vision_learner(cards,resnet50,metrics=error_rate )
    learn.fine_tune(20)
    learn.save("card_classifier.pkl")

    cleaner = ImageClassifierCleaner(learn)
    cleaner
    path = './card_classifier_update.pkl'
    cards2 = DataBlock(
        blocks= (ImageBlock, CategoryBlock),
        get_items = get_image_files,
        splitter=RandomSplitter(valid_pct=.2,seed=42),
        get_y=parent_label,
        item_tfms=Resize(128)
    ).dataloaders(path,bs=64)
