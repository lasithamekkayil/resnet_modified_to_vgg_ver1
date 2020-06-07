import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process, random_warp
import csv
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import backend as K
from glob import glob
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from PIL import Image
import random
import shutil
import math
from distutils.dir_util import copy_tree
from scipy.spatial import distance



"""
This module implements the Context Aware correlation filter based tracking algorithm with Deep learning-- MOSSE CACF 

Date: 2020-05-22

"""

class mosselas5_CACF:
    def __init__(self, args, img_path, dataset):
        # get arguments..
        self.args = args
        self.img_path = img_path
        self.dataset = dataset
        #self.gt_path = gt_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()
    
    # start to do the object tracking...
    def start_tracking(self):
        # get the image of the first frame... (read as gray scale image...)
        init_img = cv2.imread(self.frame_lists[0])
        init_img_training =init_img.astype(np.float32)
        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        f=open(os.path.join('datasets/'+ self.dataset, 'groundtruth_rect.txt'))
        #Convert the text file to a list
        gt = []
        for line in f:
            stripped_line = line. strip()
            line_list = stripped_line. split(',')
            gt. append(line_list)
        f. close()  
        #Convert the strings in the text file to a list
        for j in range(len(gt)):
            temp = gt[j]
            for i in range(0, len(temp)): 
                temp[i] = int(temp[i])
            gt[j] = temp            
        #if the target is large, lower the resolution
        #resize_img
        #step1 resize the width and height
        init_gt = gt[0] 
        width_for_accuracy_calculation = init_gt[2]
        height_for_accuracy_calculation = init_gt[3]
        target_sz = []
        window_sz=[]
        init_frame_old = init_frame
        pos, target_sz, init_frame, resize_image_done = self._resize_img(init_gt, init_frame_old)

        #perform padding to have coarse patch (bigger window size) - which is better than searching inside a fine patch
        padding = 2
        window_sz.append(np.floor(target_sz[0]*(1+padding)))
        window_sz.append(np.floor(target_sz[1]*(1+padding)))
        output_sigma_factor = 0.1
        sz_for_gaussian = np.floor(window_sz)
        out_sigma = (np.sqrt(target_sz[0]*target_sz[1]))*(output_sigma_factor)

        #compute the response map
        response_map_las = np.fft.fft2(self._gauss_label(out_sigma,window_sz))
        cos_window = np.array(np.hanning(window_sz[1]))[np.newaxis] * ((np.array(np.hanning(window_sz[0]))[np.newaxis]).T)
        
        #compute the offset for the CACF context           
        offset = np.zeros((4,2))
        offset[0,0] = -(np.floor(init_gt[3]))
        offset[1,1] = -(np.floor(init_gt[2]))
        offset[2,0] = np.floor(init_gt[3])
        offset[3,1] = np.floor(init_gt[2])

        #compute the offset for the background           
        offset_background = np.zeros((4,2))
        offset_background[0,0] = -(np.floor(window_sz[0]))
        offset_background[1,1] = -(np.floor(window_sz[1]))
        offset_background[2,0] = np.floor(window_sz[0])
        offset_background[3,1] = np.floor(window_sz[1])
          
        # start to draw the gaussian response...a
        g = response_map_las
        #lasithas patch for the first image
        patch = self._get_subwindow(init_frame, pos, window_sz)
        original_firstframe_patch = patch

        #Create train folder
        current_directory = os.getcwd()
        train_directory = os.path.join(current_directory, r'train_folder')
        os.makedirs(train_directory)
        train_foreground = os.path.join(train_directory, r'foreground')
        os.makedirs(train_foreground)
        train_background = os.path.join(train_directory, r'background')
        os.makedirs(train_background)
        #Create test folder
        test_directory = os.path.join(current_directory, r'test_folder')
        os.makedirs(test_directory)
        test_foreground = os.path.join(test_directory, r'foreground')
        os.makedirs(test_foreground)
        test_background = os.path.join(test_directory, r'background')
        os.makedirs(test_background)
        #Create evaluation folder
        evaluation_directory = os.path.join(current_directory, r'evaluation_folder')
        os.makedirs(evaluation_directory)

        #copy the contents of train_test folder to train and test folders
        #fromDirectory = "F:/combined_DL_and_MOSSECACF/datasets/Train_Test_data/" + self.dataset
        #toDirectory2 = "F:/combined_DL_and_MOSSECACF/train_folder/foreground"
        #copy_tree(fromDirectory, toDirectory2)
        
        #Deep learning stage1 starts from here######
        checking_window_sz_or_target_to_select = np.multiply(window_sz[0],window_sz[1])
        window_sz_DL =[]
        if checking_window_sz_or_target_to_select > 2000:
            window_sz_DL.append(int(np.floor(target_sz[0]*(1.5))))
            window_sz_DL.append(int(np.floor(target_sz[1]*(1.5))))
            patch_for_DL_training_1 = self._get_subwindow(init_img_training, pos, target_sz)
            patch_for_DL_training_1 = patch_for_DL_training_1.astype(np.uint8)
            cv2.imwrite(os.path.join(train_foreground , 'train_colour_img_1.jpg'), patch_for_DL_training_1)
        else:
            window_sz_DL = window_sz
        
        ##save the patch selected to train folder
        patch_for_DL_training = self._get_subwindow(init_img_training, pos, window_sz)
        patch_for_DL_training = patch_for_DL_training.astype(np.uint8)
        cv2.imwrite(os.path.join(train_foreground , 'train_colour_img.jpg'), patch_for_DL_training)

        #Generate training data by shifting 1 to 10 pixels, plus or minus from centre point
        for right_shift in range(1,10):
            wrap_right = cv2.copyMakeBorder(patch_for_DL_training,0,0,right_shift,0,cv2.BORDER_WRAP)
            cv2.imwrite(os.path.join(train_foreground , 'train_right_shifted_img'+str(right_shift)+'.jpg'), wrap_right)
        for left_shift in range(1,10):
            wrap_left = cv2.copyMakeBorder(patch_for_DL_training,0,0,0,left_shift,cv2.BORDER_WRAP)
            cv2.imwrite(os.path.join(train_foreground , 'train_left_shifted_img'+str(left_shift)+'.jpg'), wrap_left)


        #changing the dimension of patch_for_DL_training - This is performed to use the patch for data augmentation 
        patch_for_DL_training_dim_change = np.array(patch_for_DL_training)
        patch_for_DL_training_dim_change = np.expand_dims(patch_for_DL_training_dim_change, axis =0)
        ##Use imagedatagenerator and generate multiple train images from first single image patch
        datagen = ImageDataGenerator(rotation_range=8, horizontal_flip=0.5, width_shift_range=[-10,10])
        datagen.fit(patch_for_DL_training_dim_change)
        #save the generated training images
        index_img=0
        batch_size=20
        for img_batch in datagen.flow(patch_for_DL_training_dim_change, batch_size=20):
            for img in img_batch:
                cv2.imwrite(os.path.join(train_foreground , 'train_colour_img'+str(index_img)+'.jpg'), img)
                index_img = index_img+1   
            if index_img >= batch_size:
                break
        
        ##Use imagedatagenerator and generate multiple images from first single image patch for test folder/foreground
        datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=0.7, width_shift_range=[-12,12])
        datagen.fit(patch_for_DL_training_dim_change)
        #save the generated training images
        index_img_test=0
        batch_size=20
        for img_batch1 in datagen.flow(patch_for_DL_training_dim_change, batch_size=20):
            for img in img_batch1:
                cv2.imwrite(os.path.join(test_foreground , 'test_img'+str(index_img_test)+'.jpg'), img)
                index_img_test = index_img_test+1   
            if index_img_test >= batch_size:
                break
        #Generate background data and save in train_folder
        #compute immediate background with offset for the test data          
        index_background =0
        ind_back =0
        ind_back_2 = 0
        for val in range(len(offset_background)):
            patch_for_DL_background = self._get_subwindow(init_img_training, np.add(pos,offset_background[val]), window_sz)
            patch_for_DL_background =patch_for_DL_background.astype(np.uint8)
            cv2.imwrite(os.path.join(test_background , 'test_img'+str(index_background)+'.jpg'), patch_for_DL_background)
            #Generate training background data by shifting 1 to 10 pixels, plus or minus from centre point
            for right_shift in range(1,10):
                wrap_right = cv2.copyMakeBorder(patch_for_DL_background,0,0,right_shift,0,cv2.BORDER_WRAP)
                cv2.imwrite(os.path.join(train_background , 'Context_patch_shifted_img'+str(ind_back)+'.jpg'), wrap_right)
                ind_back = ind_back+1
            for left_shift in range(1,10):
                wrap_left = cv2.copyMakeBorder(patch_for_DL_background,0,0,0,left_shift,cv2.BORDER_WRAP)
                cv2.imwrite(os.path.join(train_background , 'Context_patch_left_shifted_img'+str(ind_back_2)+'.jpg'), wrap_left)
                ind_back_2 = ind_back_2+1
            index_background = index_background+1
        
        #Train the deep learning network
        # re-size all the images to this
        #for vgg16
        IMAGE_SIZE = [224, 224]
        #for resnet50
        window_sz[1] = int(np.floor(window_sz[1]))
        window_sz[0] = int(np.floor(window_sz[0]))

        #IMAGE_SIZE = [window_sz[1], window_sz[0]]
        print(type(window_sz[1]))

        train_path = 'train_folder'
        valid_path = 'test_folder'

        # add preprocessing layer to the front of VGG
        resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

        # don't train existing weights
        for layer in resnet.layers:
            layer.trainable = False
        # useful for getting number of classes
        folders = glob('train_folder/*')
        # our layers - you can add more if you want
        #x = Flatten()(incep.output)
        x=resnet.output
        x= GlobalAveragePooling2D()(x)
        x = Dense(1000, activation='relu')(x)
        #x = Dropout(0.7)(x)
        prediction = Dense(len(folders), activation='softmax')(x)
        # create a model object
        model = Model(inputs=resnet.input, outputs=prediction)
        # view the structure of the model
        model.summary()
        # tell the model what cost and optimization method to use
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory('train_folder',target_size = (224,224),batch_size =64,class_mode = 'categorical')
        test_set = test_datagen.flow_from_directory('test_folder',target_size = (224,224),batch_size =64,class_mode = 'categorical')
        # fit the model
        r = model.fit_generator(training_set,validation_data=test_set,epochs=10,steps_per_epoch=len(training_set),validation_steps=len(test_set))

        #loss
        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()
        plt.savefig('LossVal_loss')

        # accuracies
        plt.plot(r.history['accuracy'], label='train acc')
        plt.plot(r.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()
        plt.savefig('AccVal_acc')

        ####Deep learning stage 1 ends here##########

        #lasithas feature for the first image
        xf = np.fft.fft2(self._get_features(patch,cos_window))
        kf = np.multiply(np.conj(xf),xf)    
        kfn = self._get_feature_for_all_offset(init_frame, pos, offset, window_sz,cos_window, xf, width_for_accuracy_calculation, height_for_accuracy_calculation)                   
        num = np.multiply(np.conj(xf),response_map_las)
        lambda1 = .0001 
        lambda2 =20
        den = kf + lambda1 + np.multiply(lambda2,np.sum(kfn,axis=2))
        wf = np.divide(num,den)
        positions = []
        positions.append(pos) 
        final_locations =[] 
        if resize_image_done == 1:
                    width_for_accuracy_calculation = np.floor(init_gt[2])
                    height_for_accuracy_calculation = np.floor(init_gt[3]) 
        
        # start the tracking..
        for idx in range(len(self.frame_lists)):
            if idx == 0:
                model_wf =wf
                #final_locations= np.floor(init_gt)
                if resize_image_done ==1:
                    final_locations = [init_gt[0],init_gt[1], width_for_accuracy_calculation*2, height_for_accuracy_calculation*2 ]
                    file_output = open("results_CACF_resnet/"+ self.dataset+".txt", "a")
                    file_output.write(str(init_gt[0]) + "," + str(init_gt[1])+"," + str(init_gt[2])+"," + str(init_gt[3]))
                    file_output.close()
                else:
                    final_locations= np.floor(init_gt)
                    file_output = open("results_CACF_resnet/"+ self.dataset+".txt", "a")
                    file_output.write(str(init_gt[0]) + "," + str(init_gt[1])+"," + str(init_gt[2])+"," + str(init_gt[3]))
                    file_output.close()
            else:
                print(idx)
                previous_position = []
                previous_position.append(pos[0])
                previous_position.append(pos[1])
                current_frame = cv2.imread(self.frame_lists[idx])
                frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = frame_gray.astype(np.float32)
                if resize_image_done == 1:
                    current_frame = cv2.resize(current_frame, (0,0), fx=0.5, fy=0.5)
                    frame_gray = cv2.resize(frame_gray, (0,0), fx=0.5, fy=0.5)   
                patch = self._get_subwindow(frame_gray, pos, window_sz)
                zf = np.fft.fft2(self._get_features(patch,cos_window))
                response = np.fft.ifft2(np.multiply(model_wf,zf))
                response_real = response.real 
                ####target location is at the maximum response. We must take into account the fact that,
                ##if the target doesnt move, the peak will appear at the top left corner, not at the centre. 
                ##the response wrap around cycliclically
                max_value_response = np.max(response_real)
                max_pos_response =[]
                max_pos_location = []
                max_pos_location = list(np.where(response_real == max_value_response))
                # zip the 2 arrays to get the exact coordinates
                listOfCordinates = list(zip(max_pos_location[0], max_pos_location[1]))           
                max_pos_response.append(listOfCordinates[0][0])
                max_pos_response.append(listOfCordinates[0][1])               
                if (max_pos_response[0] > (len(zf)/2)):
                    max_pos_response[0] = max_pos_response[0] - len(zf)
                if (max_pos_response[1] > (len(zf[0])/2)):
                    max_pos_response[1] = max_pos_response[1] - len(zf[0])
                vert_delta = max_pos_response[0]
                horiz_delta = max_pos_response[1]
                modify_pos = []
                modify_pos.append(vert_delta)
                modify_pos.append(horiz_delta)
                pos = np.add(pos,modify_pos) 
                pos[0] = np.floor(pos[0])
                pos[1] = np.floor(pos[1]) 
                corre_output = []
                corre_output_for_nms = []
                corre_output_for_nms.append(np.floor(np.subtract(pos[1],np.divide(window_sz[1],2))))
                corre_output_for_nms.append(np.floor(np.subtract(pos[0],np.divide(window_sz[0],2))))
                corre_output_for_nms.append(np.floor(np.add(pos[1],np.divide(window_sz[1],2))))
                corre_output_for_nms.append(np.floor(np.add(pos[0],np.divide(window_sz[0],2))))
                #saving correlation output as xa,ya
                corre_output.append(np.floor(np.subtract(pos[1],np.divide(window_sz[1],2))))
                corre_output.append(np.floor(np.subtract(pos[0],np.divide(window_sz[0],2))))
                #Check the overlap value of computed patch with actual patch, if it is less than 80% run the deep learning loop
                #Deep learning stage2 starts here
                #Select and save data for evaluation for deep learning
                #save the patch selected to train folder
                file_to_save_location_from_DL = open("file_to_save_location_from_DL.txt", "a")
                pos_a = int(pos[0])
                pos_b = int(pos[1])
                file_to_save_location_from_DL.write(str(pos_a) + " " + str(pos_b))
                patch_from_correlation_for_evaluation = self._get_subwindow(current_frame, pos, window_sz_DL)
                patch_from_correlation_for_evaluation = patch_from_correlation_for_evaluation.astype(np.uint8)
                cv2.imwrite(os.path.join(evaluation_directory , 'Context_patch0.jpg'), patch_from_correlation_for_evaluation)
                results_prediction =[]
                patch_from_correlation_for_evaluation = cv2.resize(patch_from_correlation_for_evaluation, (224,224))
                img_array1 = np.array(patch_from_correlation_for_evaluation)
                img_array1 = np.expand_dims(img_array1, axis =0)
                pred1 = model.predict(img_array1)
                print(pred1)
                results_prediction.append(pred1)
                #print(type(pred1))
                #print(pred1[0])
                #print(pred1[0][1])
                if pred1[0][1] < 0.4:
                    #select all context patch around previous frame and save in the evaluation directory (all far patches too)
                    images = current_frame
                    loc1 = images.shape[0]
                    loc2 = images.shape[1]
                    overlap_row = int(np.ceil(np.divide(window_sz_DL[0],2)))
                    overlap_col = int(np.ceil(np.divide(window_sz_DL[1],2)))
                    context_patches_around =[]
                    starting_point_y = int(np.floor(np.subtract(previous_position[0],window_sz_DL[0])))
                    starting_point_x = int(np.floor(np.subtract(previous_position[1],window_sz_DL[1])))
                    ending_point_y = int(np.floor(np.add(previous_position[0],window_sz_DL[0])))
                    ending_point_x = int(np.floor(np.add(previous_position[1],window_sz_DL[1])))
                    if starting_point_x < 0:
                        starting_point_x =0
                    if ending_point_x > loc1:
                        ending_point_x = loc1
                    if starting_point_y < 0:
                        starting_point_y =0
                    if ending_point_y > loc2:
                        ending_point_y = loc2
                    if starting_point_x == 0 and ending_point_x == 0 :
                        starting_point_x =0
                        ending_point_x = window_sz_DL[1]
                    if starting_point_y ==0 and ending_point_y ==0:
                        starting_point_y =0
                        ending_point_y = window_sz_DL[0]
                    if starting_point_x == loc1 and ending_point_x == loc1 :
                        starting_point_x =np.abs(loc1 - window_sz_DL[1])
                        ending_point_x = loc1
                    if starting_point_y ==loc2 and ending_point_y ==loc2:
                        starting_point_y =np.abs(loc2 - window_sz_DL[0])
                        ending_point_y = loc2
                    ind_DL_context = 1 
                    for loc_val in range(starting_point_x,ending_point_x,overlap_col):
                        for loc_col_val in range(starting_point_y,ending_point_y,overlap_row):
                            location_x_centrepoint = np.ceil(np.add(loc_val,np.divide(window_sz_DL[1],2)))
                            location_y_centrepoint = np.ceil(np.add(loc_col_val,np.divide(window_sz_DL[0],2)))
                            location =[location_y_centrepoint, location_x_centrepoint]
                            Multiple_patched_from_context_DL = self._get_subwindow(images, location, window_sz_DL)
                            #plt.imshow(Multiple_patched_from_context_DL)
                            #plt.show()
                            Multiple_patched_from_context_DL = Multiple_patched_from_context_DL.astype(np.uint8)
                            cv2.imwrite(os.path.join(evaluation_directory , 'Context_patch'+str(ind_DL_context) +'.jpg'), Multiple_patched_from_context_DL)
                            ind_DL_context = ind_DL_context+1
                            file_to_save_location_from_DL.write("\n" + str(loc_col_val) + " " + str(loc_val))
                            Multiple_patched_from_context_DL = cv2.resize(Multiple_patched_from_context_DL, (224,224))
                            img_array = np.array(Multiple_patched_from_context_DL)
                            img_array = np.expand_dims(img_array, axis =0)
                            pred = model.predict(img_array)
                            results_prediction.append(pred)
                file_to_save_location_from_DL.close()

                #convert the list of list to one list
                flat_list = [item for sublist in results_prediction for item in sublist]
                #print(flat_list)  

                #select the predicted values of foreground images
                results_list = []
                for values in range(len(flat_list)):
                    results_list.append(flat_list[values][1])
                #print(results_list)  
                
                #find the maximum prediction value among foreground images
                max_prediction_value = max(results_list)
                print(max_prediction_value) 


                #find the position of maximum predicted value - image correspond to this position will be our target of interest
                position_max = [i for i, j in enumerate(results_list) if j == max_prediction_value]
                #print(position_max_prediction_value)

                dist = []
                if position_max[0] ==0 :
                    file_results = open("file_to_save_location_from_DL.txt", "r")
                    lines = file_results.readlines()
                    desired_position_onlyonevalue = lines[position_max[0]]
                    values_in_desired_position_onlyonevalue = desired_position_onlyonevalue.split(" ")
                    pos[0] = np.floor(int(values_in_desired_position_onlyonevalue[0]))
                    pos[1] = np.floor(int(values_in_desired_position_onlyonevalue[1]))
                    file_results.close() 
                else:
                    file_results = open("file_to_save_location_from_DL.txt", "r")
                    lines = file_results.readlines()
                    image_copy = current_frame
                    desired_tuple_inside_list = []
                    for max_pos_locations in position_max:
                        #Filterout using patches overlap with previous frame: If no overlap at all, it indicates it is a false positive - remove false positives
                        max_pos_locations_index = max_pos_locations
                        desired_position_ = lines[max_pos_locations_index]
                        position_form_of_string_ = desired_position_.rstrip()
                        values_in_desired_position_ = position_form_of_string_.split(" ")
                        final_loc_col_ = int(values_in_desired_position_[0])
                        final_loc_row_ = int(values_in_desired_position_[1])
                        starting_location_frame = []
                        starting_location_frame.append(np.floor(final_loc_row_)) 
                        starting_location_frame.append(np.floor(final_loc_col_)) 
                        previous_position_starting_point_x = int(np.floor(np.subtract(previous_position[1],np.divide(window_sz[1],2))))
                        previous_position_starting_point_y = int(np.floor(np.subtract(previous_position[0],np.divide(window_sz[0],2))))
                        previous_position_frame = []
                        previous_position_frame.append(previous_position_starting_point_x)
                        previous_position_frame.append(previous_position_starting_point_y)
                        #check for false positives and remove
                        checking_for_false_positive = self._remove_false_positive(previous_position_frame,starting_location_frame, window_sz)
                        if checking_for_false_positive ==1:
                            continue
                        desired_position_modified = []
                        desired_position_modified_y_start = np.floor(int(final_loc_col_))
                        desired_position_modified_y_end = np.floor(int(np.add(final_loc_col_,window_sz[0])))
                        desired_position_modified_x_start = np.floor(int(final_loc_row_))
                        desired_position_modified_x_end = np.floor(int(np.add(final_loc_row_,window_sz[1])))
                        desired_position_modified.append(desired_position_modified_x_start)
                        desired_position_modified.append(desired_position_modified_y_start)
                        desired_position_modified.append(desired_position_modified_x_end)
                        desired_position_modified.append(desired_position_modified_y_end)
                        list_to_tuple = tuple(desired_position_modified)
                        desired_tuple_inside_list.append(list_to_tuple)
                    boundingBoxes = np.array(desired_tuple_inside_list)
                    #Check whether the bounding boxes are empty, if empty add previous_position
                    if len(boundingBoxes) < 1:
                        #check correlation output is false positive or not
                        checking_for_false_positive_correlation = self._remove_false_positive(previous_position_frame,corre_output, window_sz)
                        if checking_for_false_positive_correlation ==0:
                            list_to_tuple_corre = tuple(corre_output_for_nms)
                            desired_tuple_inside_list.append(list_to_tuple_corre)
                        else:
                            final_loc_col_ = int(np.floor(np.subtract(previous_position[0], np.divide(window_sz[0],2))))
                            final_loc_row_ = int(np.floor(np.subtract(previous_position[1], np.divide(window_sz[1],2))))
                            desired_position_modified = []
                            desired_position_modified_y_start = np.floor(int(final_loc_col_))
                            desired_position_modified_y_end = np.floor(int(np.add(final_loc_col_,window_sz[0])))
                            desired_position_modified_x_start = np.floor(int(final_loc_row_))
                            desired_position_modified_x_end = np.floor(int(np.add(final_loc_row_,window_sz[1])))
                            desired_position_modified.append(desired_position_modified_x_start)
                            desired_position_modified.append(desired_position_modified_y_start)
                            desired_position_modified.append(desired_position_modified_x_end)
                            desired_position_modified.append(desired_position_modified_y_end)
                            list_to_tuple = tuple(desired_position_modified)
                            desired_tuple_inside_list.append(list_to_tuple)
                        boundingBoxes = np.array(desired_tuple_inside_list)
                    # perform non-maximum suppression on the bounding boxes 
                    selected_patch = self._non_max_suppression_slow(boundingBoxes, 0.3)
                    if len(selected_patch)==1:
                        pos[1] = np.floor(int(np.add(selected_patch[0][0],np.divide(window_sz[1],2))))
                        pos[0] = np.floor(int(np.add(selected_patch[0][1],np.divide(window_sz[0],2))))
                    else:
                        pos[1] = previous_position[1]
                        pos[0] = previous_position[0]
                    file_results.close()
                

                os.remove('file_to_save_location_from_DL.txt')
                # delete the contents of evaluation folder
                shutil.rmtree('evaluation_folder')
                os.makedirs(evaluation_directory)

                #deep learning stage 2 ends here - updated location of patch

                # Filter updation wrt the identified target position starts here          
                #Hi = Ai / Bi
                patch = self._get_subwindow(frame_gray, pos, window_sz)
                #plt.imshow(patch)
                #plt.show()
                xf = np.fft.fft2(self._get_features(patch,cos_window))
                kf = np.multiply(np.conj(xf),xf) 
                kfn = self._get_feature_for_all_offset(init_frame, pos, offset, window_sz,cos_window, xf, width_for_accuracy_calculation, height_for_accuracy_calculation) 
                num = np.multiply(np.conj(xf),response_map_las)
                lambda1 = .0001 
                lambda2 =1
                den = kf + lambda1 + np.multiply(lambda2,np.sum(kfn,axis=2))
                wf = np.divide(num,den)
                
                #update the filter
                interp_factor = 0.05
                model_wf = np.multiply((1-interp_factor),model_wf) + np.multiply(interp_factor,wf)
                positions = np.vstack((positions,pos))
                pos_display = []
                pos_computation_for_display0 = np.floor(np.subtract(pos[1],np.divide(width_for_accuracy_calculation,2)))
                pos_computation_for_display1 = np.floor(np.subtract(pos[0],np.divide(height_for_accuracy_calculation,2)))
                pos_display.append(pos_computation_for_display0)
                pos_display.append(pos_computation_for_display1)
                pos_display.append(width_for_accuracy_calculation)
                pos_display.append(height_for_accuracy_calculation)
                if resize_image_done == 1:
                    pos_display_resized = [2*i for i in pos_display]
                    final_locations = np.vstack((final_locations, pos_display_resized))
                    file_output = open("results_CACF_resnet/"+ self.dataset+".txt", "a")
                    file_output.write("\n" + str(pos_display_resized[0]) + "," + str(pos_display_resized[1])+"," + str(pos_display_resized[2])+"," + str(pos_display_resized[3]))
                    file_output.close()
                else:
                    final_locations = np.vstack((final_locations, pos_display))
                    file_output = open("results_CACF_resnet/"+ self.dataset+".txt", "a")
                    file_output.write("\n" + str(pos_display[0]) + "," + str(pos_display[1])+"," + str(pos_display[2])+"," + str(pos_display[3]))
                    file_output.close()

                
                
                # visualize the tracking process...
                # represents the top left corner of rectangle 
                start_point = (int(pos_display[0]), int(pos_display[1])) 
                # Ending coordinate, here (220, 220) 
                # represents the bottom right corner of rectangle 
                endpoint0 = int(pos_display[0]+pos_display[2])
                endpoint1 = int(pos_display[1]+pos_display[3])
                end_point = (endpoint0, endpoint1) 
                # Blue color in BGR 
                color = (255, 0, 0) 
                # Line thickness of 2 px 
                thickness = 2
                # Using cv2.rectangle() method 
                # Draw a rectangle with blue line borders of thickness of 2 px 
                cv2.rectangle(current_frame, start_point, end_point, color, thickness) 
                #cv2.rectangle(current_frame, (pos_display[0], pos_display[1]), (pos_display[0]+pos_display[2], pos_display[1]+pos_display[3]), (255, 0, 0), 2)
                cv2.imshow('demo', current_frame)
                cv2.waitKey(100)
                # if record... save the frames..
                if self.args.record:
                    frame_path = 'record_frames/' + self.img_path.split('/')[1] + '/'
                    if not os.path.exists(frame_path):
                        os.mkdir(frame_path)
                    cv2.imwrite(frame_path + str(idx).zfill(5) + '.png', current_frame)
                        #Delete all the evaluation folder
        shutil.rmtree('train_folder') 
        shutil.rmtree('test_folder')
        #shutil.rmtree('evaluation_folder')        
        return [final_locations,gt]  
                          
    
    def iou_computation_CACFmethod(self, val):  
        #Accuracy Computation
        gt_row =[]
        mosse_out =[]
        acc_val =[]
        overall_iou = []
        iou_each_frame =[]
        ground_values = val[1]
        #mosseCACF_values = val[1]
        mosseCACF_values = val[0]
        for i in range(0, len(ground_values)): 
            #print(len(ground_values))
            gt_row = ground_values[i]
            mosse_out = list(mosseCACF_values[i])
            r1 =[]
            r2 =[]
            c1 = []
            c2 =[]
            ci = []
            #r1 = [(gt_row[1]+np.divide(gt_row[3],2)), (gt_row[0]+np.divide(gt_row[2],2)), gt_row[2], gt_row[3] ]
            r1 = [(gt_row[1]+np.divide(gt_row[3],2)), (gt_row[0]+np.divide(gt_row[2],2)), mosse_out[2], mosse_out[3] ]
            r2 = [(mosse_out[1]+np.divide(mosse_out[3],2)), (mosse_out[0]+np.divide(mosse_out[2],2)), mosse_out[2], mosse_out[3] ]
            c1 = [r1[0],r1[1],r1[2]+r1[0],r1[3]+r1[1]]
            c2 = [r2[0],r2[1],r2[2]+r2[0],r2[3]+r2[1]]
            #compute coordinates of maximum intersection
            ci = [max(c1[0], c2[0]), max(c1[1],c2[1]), min(c1[2],c2[2]), min(c1[3],c2[3])]
            wi = ci[2]-ci[0]
            hi = ci[3]-ci[1]
            ai = np.multiply(wi,hi)
            au = (np.multiply(r1[2],r1[3]) + np.multiply(r2[2],r2[3])) - ai
            iou_frame_wise = np.divide(ai,au)
            iou_each_frame.append(np.abs(iou_frame_wise))             
        overall_iou = np.mean(iou_each_frame)           
        return overall_iou 

    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi
    
        # get the ground-truth gaussian reponse...  
    #lasithas gaussian response CACF
    def _gauss_label(self,out_sigma,window_sz):
        #eveluate a gaussian with the peak at the centre element
        #col, row = np.meshgrid((np.arange(window_sz[1]))-np.floor(window_sz[1]/2), (np.arange(window_sz[0])--np.floor(window_sz[0]/2)))
        col, row = np.meshgrid((np.arange(window_sz[1])), (np.arange(window_sz[0])))
        updated_col = np.subtract(col,np.floor(window_sz[1]/2))
        updated_row = np.subtract(row,np.floor(window_sz[0]/2))
        intermediate_step1 = np.square(updated_row)
        intermediate_step2 = np.square(updated_col)
        intermediate_step3 = np.add(intermediate_step1, intermediate_step2)
        intermediate_step4 = np.divide(-0.5, np.square(out_sigma))
        intermediate_step5 = (intermediate_step4)* intermediate_step3
        labels = np.exp(intermediate_step5)
        # move the peak to the topleft with wrap around
        val0 = int(np.floor((window_sz[0]/2)))
        val1 = int(np.floor((window_sz[1]/2)))
        x = np.roll(labels, -val0, axis=0) # up
        x1 = np.roll(x, -val1, axis=1) # left
        assert(x1[0][0] ==1)
        return x1

    # it will extract the image list 
    def _get_img_lists(self, img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list
    
    # it will get the first ground truth of the video..
    def _get_init_ground_truth(self, img_path):
        gt_path = os.path.join(img_path, 'groundtruth.txt')
        with open(gt_path, 'r') as f:
            # just read the first frame...
            line = f.readline()
            gt_pos = line.split(',')

        return [float(element) for element in gt_pos]
     
    def _get_subwindow(self, init_frame, pos, window_sz):
        init_frame = init_frame.astype(int)
        height_of_image = init_frame.shape[0]
        width_of_image = init_frame.shape[1]
        xs = np.round(pos[1]) + np.arange(0,window_sz[1],1) - np.floor(np.divide(window_sz[1],2))  
        ys = np.round(pos[0]) + np.arange(0,window_sz[0],1) - np.floor(np.divide(window_sz[0],2))
        #Checkout for out of bound corordinates, and set them to the values at the borders
        padding_left = 0
        padding_left = [padding_left+1 for xs_val in xs if xs_val<0]
        xs = [0 if xs_val<0 else xs_val for xs_val in xs]
        padding_top = 0
        padding_top= [padding_top +1 for ys_val in ys if ys_val<0]
        ys = [0 if ys_val<0 else ys_val for ys_val in ys]  
        padding_right = 0
        padding_right = [padding_right+1 for xs_val in xs if xs_val>width_of_image ]
        xs = [(width_of_image) if xs_val>width_of_image else xs_val for xs_val in xs]
        padding_bottom =0
        padding_bottom = [padding_bottom+1 for ys_val in ys if ys_val>height_of_image ]   
        ys = [(height_of_image) if ys_val>height_of_image else ys_val for ys_val in ys]
        xs = [int(x) for x in xs]
        ys = [int(y) for y in ys]
        height_ = window_sz[0]
        width_ = window_sz[1]
        top_left_location_y = np.floor(pos[0])-np.floor(np.divide(window_sz[0],2))
        top_left_location_x = np.floor(pos[1])-np.floor(np.divide(window_sz[1],2))
        patch_starting_point_y = ys[0]
        patch_starting_point_x = xs[0]
        patch_end_point_y = ys[len(ys)-1]
        patch_end_point_x = xs[len(xs)-1]
        
        patch = init_frame[patch_starting_point_y:patch_end_point_y, patch_starting_point_x:patch_end_point_x]            
        patch = cv2.copyMakeBorder(patch,len(padding_top), len(padding_bottom), len(padding_left), len(padding_right), cv2.BORDER_REPLICATE)
        if patch.shape[0] != window_sz[0]:
            extra_padding_bottom = window_sz[0] - patch.shape[0]
        else:
            extra_padding_bottom = 0 
        if patch.shape[1] !=0:
            extra_padding_right = window_sz[1] - patch.shape[1]
        else:
            extra_padding_right =0
            
        patch = cv2.copyMakeBorder(patch,0, int(extra_padding_bottom), 0, int(extra_padding_right), cv2.BORDER_REPLICATE)    
        return patch 
      
    def _get_subwindow_old_function(self, init_frame, pos, window_sz):
        frame = init_frame
        height = int(window_sz[0]) 
        width = int(window_sz[1]) 
        top_left_location_y = int(np.floor(np.subtract(pos[0]),(np.divide(window_sz[0],2))))
        top_left_location_x = int(np.floor(np.subtract(pos[1]),np.divide(window_sz[1],2)))
        #check last values are going out of boundary
        if (top_left_location_y < 0):
            top_left_location_y = 0
        if (top_left_location_x < 0):
            top_left_location_x = 0                  
        patch = frame[top_left_location_y:top_left_location_y+height, top_left_location_x:top_left_location_x+width]
        return patch
    
    def _get_features(self, patch, cos_window):
        x = np.double(patch)/255
        x = x - np.mean(x)

        if cos_window.any():
            x = np.multiply(x,cos_window) 
            return x
    
    def _resize_img(self, init_gt, init_frame_old):
        if ((np.sqrt(np.multiply(init_gt[2],init_gt[3]))) >= 100):
            #resize the image
            pos = []
            #pos =[(init_gt[3]/2) + init_gt[1], (init_gt[2]/2) + init_gt[0]]
            pos_intermediate = [np.add(np.divide(init_gt[3],2),init_gt[1]),np.add(np.divide(init_gt[2],2),init_gt[0])]
            pos =[np.floor(np.divide(pos_intermediate[0],2)), np.floor(np.divide(pos_intermediate[1],2))]
            init_gt_1 = np.abs(np.divide(init_gt[2],2))
            init_gt_2 = np.abs(np.divide(init_gt[3],2))
            target_sz = [np.floor(init_gt_2), np.floor(init_gt_1)]
            resize_val1 = int(np.floor(np.divide(init_frame_old.shape[1],2)))
            resize_val2 = int(np.floor(np.divide(init_frame_old.shape[0],2)))
            #init_frame = np.resize(init_frame_old, (resize_val1, resize_val2)) 
            resize_image_done =1
            init_frame = cv2.resize(init_frame_old, dsize=(resize_val1, resize_val2), interpolation=cv2.INTER_CUBIC)    
        else:
            pos = []
            pos = [np.add(np.divide(init_gt[3],2),init_gt[1]),np.add(np.divide(init_gt[2],2),init_gt[0])]
            #pos =[(init_gt[3]/2) + init_gt[1], (init_gt[2]/2) + init_gt[0]]
            target_sz=[init_gt[3], init_gt[2]]
            init_frame = init_frame_old
            resize_image_done =0
        return pos, target_sz, init_frame, resize_image_done

    def _without_resize_img(self, init_gt, init_frame_old):
        pos = []
        target_sz = []
        init_frame =init_frame_old
        pos =[np.add(np.divide(init_gt[3],2), init_gt[1]), np.add(np.divide(init_gt[2],2),init_gt[0])]
        target_sz=[init_gt[3], init_gt[2]]
        init_frame = init_frame_old
        return pos, target_sz, init_frame
    
    def _get_feature_for_all_offset(self, init_frame, pos, offset, window_sz, cos_window,xf,width_for_accuracy_calculation, height_for_accuracy_calculation):
    #lasitha - computing stacked features for all offset locations
        for val in range(len(offset)):
            patch = self._get_subwindow(init_frame, np.add(pos,offset[val]), window_sz)
            xfn = np.fft.fft2(self._get_features(patch,cos_window))
            out_feature = np.multiply(np.conj(xfn), xfn)
            if val ==0:
                temp = out_feature
            elif val ==1:
                temp1 = out_feature
            elif val ==2:
                temp2 = out_feature
            else:
                temp3 = out_feature
                kfn = np.dstack((temp,temp1,temp2,temp3))
    
                return kfn

    def _computation_input_frame_without_target(self, initalimg, initialgt, window_sz): 
        images_training = initalimg
        shape_x = images_training.shape[0]
        shape_y = images_training.shape[1]
        pos0 = int(np.floor(initialgt[1]))-30
        pos1 = int(np.floor(initialgt[0]))-30
        if pos0<0 :
            pos0 = 0
        if pos1<0 :
            pos1 = 0
        window_sz0 = int(np.floor(window_sz[0]))+30
        window_sz1 = int(np.floor(window_sz[1]))+30
        if window_sz0 > images_training.shape[0] :
            window_sz0 = images_training.shape[0]
        if window_sz1 > images_training.shape[1] :
            window_sz1 = images_training.shape[1]
        images_training[pos0:pos0+window_sz0,pos1:pos1+window_sz1] = np.zeros((window_sz0,window_sz1,3))
        #imgplot = plt.imshow(images_training)
        #plt.show() 
        return images_training

    def _non_max_suppression_slow(self, boxes, overlapThresh):
	    if len(boxes) == 0:
		    return []
	    pick = []
	    x1 = boxes[:,0]
	    y1 = boxes[:,1]
	    x2 = boxes[:,2]
	    y2 = boxes[:,3]
	    area = (x2 - x1 + 1) * (y2 - y1 + 1)
	    idxs = np.argsort(y2)
	    while len(idxs) > 0:
		    last = len(idxs) - 1
		    i = idxs[last]
		    pick.append(i)
		    suppress = [last]
		    for pos in range(0, last):
			    j = idxs[pos]
			    xx1 = max(x1[i], x1[j])
			    yy1 = max(y1[i], y1[j])
			    xx2 = min(x2[i], x2[j])
			    yy2 = min(y2[i], y2[j])
			    w = max(0, xx2 - xx1 + 1)
			    h = max(0, yy2 - yy1 + 1)
			    overlap = float(w * h) / area[j]
			    if overlap > overlapThresh:
				    suppress.append(pos)
		    idxs = np.delete(idxs, suppress)
	    return boxes[pick]   
    
    def _compute_best_patch(self, selected_patch, previous_position):
        previous_xywh = []
        previous_xywh.append(previous_position[0])
        previous_xywh.append(previous_position[1])
        all_ious = []
        identified_patch = []
        selected_patch_after_iou =[]
        for selected_patches in selected_patch:    
            selected_patches_width = selected_patches[0][2]- selected_patches[0][0]
            selected_patches_height = selected_patches[0][3]- selected_patches[0][1]
            input_xywh = []
            previous_xywh.append(selected_patches_width)
            previous_xywh.append(selected_patches_height)
            input_xywh.append(selected_patches[0][0])
            input_xywh.append(selected_patches[0][1])
            input_xywh.append(selected_patches_width)
            input_xywh.append(selected_patches_height)
            xA1 = max(previous_xywh[0], input_xywh[0])
            yA1 = max(previous_xywh[1], input_xywh[1])
            xB1 = min(previous_xywh[2], input_xywh[2])
            yB1 = min(previous_xywh[3], input_xywh[3])
            interArea = max(0, xB1 - xA1 + 1) * max(0, yB1 - yA1 + 1)
            boxAArea = previous_xywh[2]*previous_xywh[3]
            boxBArea = input_xywh[2]*input_xywh[3]
            iou = interArea / float(boxAArea + boxBArea - interArea)
            all_ious.append(iou)
        max_value = max(all_ious)
        loc_max_valuee = max_value.index(max(max_value))
        identified_patch = selected_patches[loc_max_valuee]
        selected_patch_after_iou.append(identified_patch[1])
        selected_patch_after_iou.append(identified_patch[0])
        return selected_patch_after_iou

    def _remove_false_positive(self, previous_position,centre_location_frame, window_sz):
        xA1 = max(previous_position[0], centre_location_frame[0])
        yA1 = max(previous_position[1], centre_location_frame[1])
        xB1 = min(np.add(previous_position[0],window_sz[1]), np.add(centre_location_frame[0],window_sz[1]))
        yB1 = min(np.add(previous_position[1],window_sz[0]), np.add(centre_location_frame[1],window_sz[0]))
        interArea = max(0, xB1 - xA1 + 1) * max(0, yB1 - yA1 + 1)
        boxAArea = window_sz[0]*window_sz[1]
        boxBArea = window_sz[0]*window_sz[1]
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou > 0.2:
            false_pos =0
        else:
            false_pos =1
        return false_pos