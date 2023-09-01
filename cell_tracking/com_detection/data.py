import json
import csv
import numpy as np
from PIL import Image
from math import ceil
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint
import glob
import os
from skimage.transform import resize
from functools import partial
import multiprocessing as mp
from scipy.stats import multivariate_normal

def make_peak(com,pmap_size,cov):
    com = (com[0]*pmap_size[0],com[1]*pmap_size[1])
    pos = np.stack(np.meshgrid(
            np.arange(1,pmap_size[0]+1),
            np.arange(1,pmap_size[1]+1)),axis=-1)
    peak = multivariate_normal.pdf(pos,mean=(com[0],com[1]),cov=cov)
    return peak

class CellData:
    '''
    Class for processing and generating tiles of cell image data
    '''

    ###Class initialization###
    def __init__(self,img_paths,max_coms,coms_paths=None,size_dict=None,
                 batch_size=32,sample_dim=128,overlap=0.25,clip=0.0,
                 target_type='p_map',shuffle=True,random_train=None):
        #(List) List of paths to the dataset images
        self.img_paths = img_paths
        #(Int) Number of COMs to return, MUST BE CONSISTENT WITH NETWORK
        self.out_len = max_coms
        #(List) List of paths to the COMs files if not with img_paths
        self.coms_paths = coms_paths
        #(Dict) Mean size of sells in each image, used for resizing tiles
        self.size_dict = size_dict

        ##Kwargs
        #Number of samples per batch (default 32)
        self.batch_size = batch_size
        #Dimension of image tile samples (PxP square, default 128x128)
        self.sample_dim = sample_dim
        #Percentage of overlapping pixels between samples (default 25%)
        self.overlap = overlap
        #Percentage of border to ignore or add when getting COMs (default 0%)
        self.clip = clip
        #Which format targets to return (defaults to "p_map")
        #Options: (x,y) coords or likelihood map consisting of gaussians
        assert target_type in ['p_map','coords'],"Invalid 'target_type'."
        self.target_type = target_type
        #Whether or not to shuffle samples during training (default True)
        self.shuffle = shuffle
        #How many random crops to use per train image (default None; tile image)
        self.random_train = random_train

        #Augmentation used during training
        self.aug = iaa.SomeOf((0, 3),
            [iaa.Fliplr(),
             iaa.Flipud(),
             iaa.OneOf([iaa.Affine(rotate=90),
                        iaa.Affine(rotate=180),
                        iaa.Affine(rotate=270)]),
             iaa.Affine(shear=(-16, 16)),
             iaa.GaussianBlur(sigma=(0.5, 5.0))])

        #Get corresponding lists of COMs and masks during initialization
        print("{} images in dataset.".format(len(img_paths)))

        if self.coms_paths is not None:
            print("Using {} manually provided COM files".format(
                len(self.coms_paths)))

        else:
            self.coms_paths = []
            missing_count = 0
            for p in img_paths:
                filename = os.path.splitext(os.path.basename(p))[0] + "_COMs.csv"
                com_path = os.path.dirname(os.path.dirname(p))
                com_path = os.path.join(com_path,"coms",filename)
                if os.path.isfile(com_path):
                    self.coms_paths.append(com_path)
                else:
                    missing_count += 1
            print("Found {} COM files, {} images have no corresponding COM file.".format(
                    len(self.coms_paths),missing_count))

        self.mask_paths = []
        missing_count = 0
        for p in img_paths:
            mask_dir = os.path.dirname(os.path.dirname(p))
            mask_dir = os.path.join(mask_dir,"masks")
            if os.path.isdir(mask_dir):
                masks_list = glob.glob(mask_dir+'/*.png')
                if len(masks_list) == 0:
                    raise ValueError("No masks found in %s",mask_dir)
                self.mask_paths.append(masks_list)
            else:
                missing_count += 1
        print("Found {} mask directories, {} images have no corresponding masks.".format(
                len(self.mask_paths),missing_count))

    ###Class methods###
    def read_json(self,path):
        '''Reads the default-formatted JSON file and returns COMs
        '''
        data = json.load(open(path,'r'))
        x = [dp['points'][0][0] for dp in data['shapes']]
        y = [dp['points'][0][1] for dp in data['shapes']]
        return list(zip(x,y))

    def read_csv(self,path):
        '''Reads a CSV file consisting of (row,col) locations and returns COMs
        '''
        x = []
        y = []
        with open(path,'r',newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                x.append(int(row[1]))
                y.append(int(row[0]))
        return list(zip(x,y))

    def get_iters(self):
        '''Creates iterators for tile generation based on class args
        '''
        if self.random_train:
            steps = ceil((len(self.img_paths)*
                          self.random_train)/self.batch_size)
            return steps

        else:
            iters = []
            for path in self.img_paths:
                if self.size_dict:
                    tile_size = int(5*self.size_dict[path])
                else:
                    tile_size = self.sample_dim

                image = Image.open(path)
                width,height = image.size
                w_samp = list(range(0,width+1,
                               tile_size-int(self.overlap*tile_size)))
                h_samp = list(range(0,height+1,
                               tile_size-int(self.overlap*tile_size)))

                w_iter,h_iter = np.meshgrid(w_samp,h_samp)
                w_iter,h_iter = w_iter.ravel(),h_iter.ravel()

                shuf = np.arange(len(w_iter))
                np.random.seed(1)
                np.random.shuffle(shuf)
                w_iter = w_iter[shuf]
                h_iter = h_iter[shuf]

                iters.append((w_iter,h_iter))

            total = [x for l in iters for x in l[0]]
            steps = ceil(len(total)/self.batch_size)
            return iters,steps
        
    def parallel_pmap(self,cents,pixel_cov=2):
        map_batch = []
        peaks = partial(make_peak,
                    pmap_size=(self.sample_dim,self.sample_dim),cov=pixel_cov)
        with mp.Pool(8) as p:
            peak_stack = p.map(peaks,cents,10)
        p_map = np.stack(peak_stack,axis=-1)
        p_map = p_map.max(axis=-1)
        p_map /= p_map.max()
        p_map = np.expand_dims(p_map,axis=-1)

        return p_map

    def __process_batch(self,image_batch,centroid_batch,augment):
        '''Hidden method for uniformly processing a batch before it is yielded
        '''
        image_batch = np.array(image_batch)
        if augment: #augment images and COMs if desired
            batch = self.aug(images=image_batch,
                             keypoints=centroid_batch)
            (image_batch,keypoint_batch) = batch
            centroid_batch = []
            for cents in keypoint_batch:
                cents = [(k.x/self.sample_dim,k.y/self.sample_dim,1)
                         for k in cents]
                centroid_batch.append(cents)

        if self.target_type == 'coords':
            for cents in centroid_batch:
                if cents == []:
                    cents.append((0,0,1))
                while len(cents) < self.out_len:
                    cents.append((0,0,0))
            centroid_batch = np.array(centroid_batch,dtype=np.float32)
            np.clip(centroid_batch,0.0,1.0,out=centroid_batch)

        elif self.target_type == 'p_map':
            maps = []
            for cents in centroid_batch:
                if not cents:
                    p_map = np.zeros((self.sample_dim,self.sample_dim,1))
                    map_batch.append(p_map)
                else:
                    map_batch.append(parallel_pmap(cents))
            centroid_batch = np.array(map_batch,dtype=np.float32)

        image_batch = image_batch.astype(np.float32)/2**8
        return image_batch,centroid_batch

    def __generator(self,split,rand,
                    read=read_json,shuffle=True,augment=True):
        '''
        The base function for data batch generation.
        args:
            ONLY ONE OF THE FOLLOWING CAN EVALUATE TO TRUE
            split: either train or val from get_splits
            rand: number of tiles to randomly sample from each image
        kwargs:
            read: the COMs file reader to use (default read_csv)
            shuffle: whether or not to shuffle between epochs (default True)
            augment: whether or not to augment images (default True)
        '''
        #Checking that both arguments are not called
        if split and rand:
            raise ValueError("Generator called both 'split' and 'rand'")

        while True:
            #Initialize batch lists
            image_batch = []
            centroid_batch = []
            #Loop over files in lists
            for file_idx,(img_path,com_path) in enumerate(zip(
                         self.img_paths,self.coms_paths)):
                targets = read(self,com_path)
                image = Image.open(img_path)
                
                if self.size_dict:
                    tile_size = int(5*self.size_dict[img_path])
                else:
                    tile_size = self.sample_dim

                if split:
                    w_iter,h_iter = split[file_idx]
                elif rand:
                    w_iter = np.random.randint(image.size[0]-tile_size,
                                               size=rand)
                    h_iter = np.random.randint(image.size[1]-tile_size,
                                               size=rand)

                if shuffle and split:
                    np.random.shuffle(w_iter)
                    np.random.shuffle(h_iter)

                #Loop over tiles in file
                for left,upper in zip(w_iter,h_iter):

                    successful = False
                    #Get tile boundaries from iters
                    right = left+tile_size
                    lower = upper+tile_size

                    while not successful:
                        #Get centroids for tile from targets
                        if augment: #use Keypoint
                            scl = self.sample_dim/tile_size
                            centroids = [Keypoint(x=int((pt[0]-left)*scl),
                                                  y=int((pt[1]-upper)*scl))
                                     for pt in targets if
                            ((pt[0] >= left+int(self.clip*tile_size) and
                              pt[0] <= right-int(self.clip*tile_size)) and
                             (pt[1] >= upper+int(self.clip*tile_size) and
                              pt[1] <= lower-int(self.clip*tile_size)))]
                        else: #don't mess with Keypoint
                            centroids = [((pt[0]-left)/tile_size,
                                          (pt[1]-upper)/tile_size,1) 
                                     for pt in targets if
                            ((pt[0] >= left+int(self.clip*tile_size) and
                              pt[0] <= right-int(self.clip*tile_size)) and
                             (pt[1] >= upper+int(self.clip*tile_size) and
                              pt[1] <= lower-int(self.clip*tile_size)))]

                        #Check for max number of COMs
                        if len(centroids) <= self.out_len:
                            successful = True
                        elif rand:
                            left = np.random.randint(image.size[0]-tile_size)
                            upper = np.random.randint(image.size[1]-tile_size)
                            right = left+tile_size
                            lower = upper+tile_size
                        else:
                            e = ("Image tile from file " + img_path +
                            " located at (" + str(left) + "," + str(upper) +
                            ") contains " + str(len(centroids)-self.out_len) +
                            " too many COMs.")
                            raise ValueError(e)

                    centroid_batch.append(centroids)

                    #Get image tile
                    tile = image.crop((left,upper,right,lower))
                    if self.size_dict:
                        tile = tile.resize((self.sample_dim,self.sample_dim),
                                           resample=Image.BICUBIC)
                    tile = np.array(tile)[...,None]
                    image_batch.append(tile)

                    #Process and yield batches as arrays, reinitialize lists
                    if len(image_batch) == self.batch_size:
                        image_batch,centroid_batch = self.__process_batch(
                            image_batch,centroid_batch,augment)
                        yield image_batch,centroid_batch
                        image_batch = []
                        centroid_batch = []

            #Process and yield final batch if necessary
            if len(image_batch) != 0:
                image_batch,centroid_batch = self.__process_batch(
                    image_batch,centroid_batch,augment)
                yield image_batch,centroid_batch
                image_batch = []
                centroid_batch = []

    def train_generator(self,split):
        '''Wrapper for base generator. To be used for training data.
        '''
        if self.random_train:
            if split != None:
                e = "The CellData object was called with random_train but the \
                        train_generator was called with split not equal to \
                        'None'. Please select only one of the two train modes."
                raise ValueError(e)
            return self.__generator(split=None,rand=self.random_train,
                                    shuffle=False)
        else:
            return self.__generator(split,rand=None)

    def val_generator(self,split):
        '''Wrapper for base generator. To be used for validation data.
        '''
        if self.random_train:
            e = "val_generator should not be used on a CellData object called \
                    with random_train as it will result in data leakage."
            raise ValueError(e)
        return self.__generator(split,rand=None,shuffle=False,augment=False)

    def get_image_mask(self,mask_paths):
        '''Make a single, integer indexed cell mask array from multiple
           single-cell mask image files
        '''
        masks = []
        for i,path in enumerate(mask_paths):
            im = Image.open(path)
            mask = np.array(im)
            masks.append(np.where(mask > 0,np.full_like(mask,i+1),0))
        mask_array = np.stack(masks,axis=-1)
        mask_array = mask_array.max(axis=-1)
        return mask_array

    def test_generator(self,split,
                      read=read_json):
        '''Generator method for test data. Utilizes masks to get metrics.
        '''
        if self.random_train:
            e = "test_generator should not be used on a CellData object \
                    called with random_train as it will result in leakage."
            raise ValueError(e)
        #Loop over files in lists
        for file_idx,img_path in enumerate(self.img_paths):
            targets = read(self,self.coms_paths[file_idx])
            mask = self.get_image_mask(self.mask_paths[file_idx])
            image = Image.open(img_path)
            if self.size_dict:
                tile_size = int(5*self.size_dict[img_path])
            else:
                tile_size = self.sample_dim

            w_iter,h_iter = split[file_idx]
            #Loop over tiles in file
            for left,upper in zip(w_iter,h_iter):
                #Get tile boundaries from iters
                right = left+tile_size
                lower = upper+tile_size

                #Get centroids for tile from targets
                abs_cents = []
                for pt in targets:
                    if ((pt[0] >= left and pt[0] <= right) and
                        (pt[1] >= upper and pt[1] <= lower)):
                        abs_cents.append((pt[0],pt[1],1))

                #Get masks for tile from mask array using centroids
                mask_idx = []
                for cent in abs_cents:
                    cent_x = round(cent[0])
                    cent_y = round(cent[1])
                    mask_val = mask[cent_y,cent_x]
                    if mask_val == 0:
                        continue
                    else:
                        mask_idx.append(mask_val)
                crop_mask = np.zeros(mask.shape,dtype=np.uint16)
                for obj in mask_idx:
                    crop_mask[mask==obj] = obj
                crop_mask = crop_mask[upper:lower,left:right]
                crop_mask = resize(crop_mask,(self.sample_dim,self.sample_dim),
                                order=0,preserve_range=True,anti_aliasing=False)

                #Get image tile
                tile = image.crop((left,upper,right,lower))
                tile = tile.resize((self.sample_dim,self.sample_dim),
                                   resample=Image.BICUBIC)
                tile = np.array(tile)[:,:,0:3]

                #Process and yield
                centroids = [((x-left)/tile_size,(y-upper)/tile_size,m) for
                             (x,y,m) in abs_cents]
                if centroids == []:
                    centroids.append((0,0,1))
                while len(centroids) < self.out_len:
                    centroids.append((0,0,0))
                centroids = np.array(centroids,dtype=np.float32)
                tile = tile.astype(np.float32)/2**8
                yield (tile[np.newaxis],centroids[np.newaxis],crop_mask,mask_idx)

    def infer_generator(self,split):
        '''Generator method to be used for inference and other evaluation that
           does not utilze ground truth COMs or masks. Only yields image tiles.
        '''
        if self.random_train:
            e = "infer_generator should not be used on a CellData object \
                    called with random_train as it will result in leakage."
            raise ValueError(e)
        #Loop over files in lists
        for file_idx,img_path in enumerate(self.img_paths):
            #Returns all tiles from an image as a single batch
            image_batch = []
            loc_batch = []
            image = Image.open(img_path)
            image_size = image.size
            if self.size_dict:
                tile_size = int(5*self.size_dict[img_path])
            else:
                tile_size = self.sample_dim

            w_iter,h_iter = split[file_idx]
            #Loop over tiles in file
            for left,upper in zip(w_iter,h_iter):
                #Get tile boundaries from iters
                right = left+tile_size
                lower = upper+tile_size
                loc_batch.append((left,upper))

                tile = image.crop((left,upper,right,lower))
                if self.size_dict:
                    tile = tile.resize((self.sample_dim,self.sample_dim),
                                   resample=Image.BICUBIC)
                tile = np.array(tile)[...,None]
                tile = tile.astype(np.float32)/2**8
                image_batch.append(tile)
            yield np.array(image_batch),loc_batch,tile_size,image_size
