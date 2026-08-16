
# ================= Import Required Modules =================
import os
import cv2
import glob
import skimage
import logging  
import pydicom
import numpy as np
import pandas as pd  
from PIL import Image
from scipy import stats 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ==========================================================

class MammographyPreprocessor:
    def __init__(self, reference_image):
        
        self.reference_image = reference_image
        reference_image = cv2.normalize(reference_image,dst=None, alpha=0, beta=4095, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U) 
        self.ref_mean = np.mean(reference_image[reference_image > 0.0])
        self.ref_std = np.std(reference_image[reference_image > 0.0])

    def most_frequent_gray_value(self, img):
        flat_gray_values = img.flatten()
        most_frequent_value = stats.mode(flat_gray_values)
        return most_frequent_value.mode[0], most_frequent_value.count[0]

    def minimize_background(self, image):
        try:
            y_coord, x_coord = np.where(image != 0)
            y_min, y_max = min(y_coord), max(y_coord)
            x_min, x_max = min(x_coord), max(x_coord)
            return image[y_min:y_max, x_min:x_max]
        except ValueError:
            print('Not Found')
            return image

    def nipple_marker_removal(self, image, max_val):
        try:
            y_coord, x_coord = np.where(image == max_val)
            y_min, y_max = min(y_coord), max(y_coord)
            x_min, x_max = min(x_coord), max(x_coord)
            image[y_min - 5:y_max + 5, x_min - 5:x_max + 5] = 0
            return image
        except ValueError:
            print('Not Found')
            return image


    
    def annotation_removal(self, image ,img_ds):  
        try:  
            image = image.copy()
            # Check for the existence of the laterality tag before accessing its value  
            if (0x0020, 0x0062) in img_ds:  
                laterality = img_ds[0x0020, 0x0062].value  
            else:  
                print('Laterality tag (0020, 0062) not found. Skipping annotation removal.')  
                return image  # Return the original image if the tag is not found  
    
            # Proceed with annotation removal based on laterality  
            if laterality == 'R':  
                quarter = image[1:image.shape[0] // 2, 1:image.shape[1] // 2]  
            else:  
                quarter = image[1:image.shape[0] // 2, image.shape[1] // 2:]  
    
            M = quarter.max()  
            if M != 0:  
                y_coord, x_coord = np.where(quarter == M)  
                y_min, y_max = min(y_coord), max(y_coord)  
                x_min, x_max = min(x_coord), max(x_coord)  
                quarter[y_min - 5:y_max + 5, x_min - 5:x_max + 5] = 0  
    
            # Update the image with the modified quarter  
            if laterality == 'R':  
                image[1:image.shape[0] // 2, 1:image.shape[1] // 2] = quarter  
            else:  
                image[1:image.shape[0] // 2, image.shape[1] // 2:] = quarter  
    
            return image  

        except (ValueError, KeyError) as e:  
            print(f'Error during annotation removal: {e}')  
            return image
        
    def marker_Removal(self,image):
        try:
            y_coord,x_coord = np.where(image >= 4093)
            if len(y_coord)!=0:
                print('Marker Found')
                image[y_coord,x_coord]=int(0)
    
    
            else:
               print('Not Found Marker')
    
            return image 
        except:
            print('Not Found')
            return image

        
        
    def redundant_removal(self, img, img_ds):
        
        if (0x0020, 0x0062) in img_ds:  
            laterality = img_ds[0x0020, 0x0062].value 
        else:
            tenth_column = img[:,10]
            # Counting occurrences  
            count = np.count_nonzero(tenth_column == 0) 
            
            if   count > (img.shape[0]/2):
                laterality = 'R'
            else :
                laterality = 'L'
            
        n_rows_of_result, n_cols_of_result = img.shape
        
        # Remove redundant anatomy from the chest wall
        for r in range(n_rows_of_result - 1, n_rows_of_result // 2, -2):
            if np.sum(img[r, :] == 0) > n_cols_of_result / 1.5:
                img = img[:r-1, :]
            else:
                break

        for r in range(0, n_rows_of_result // 2, 2):
            if np.sum(img[r, :] == 0) > n_cols_of_result / 1.25:
                img = img[r:, :]
            else:
                break

        # Remove nipple area based on laterality
        for r in range(0, n_cols_of_result // 2):
            if laterality == 'R':
                if np.sum(img[:, r] == 0) > n_rows_of_result / 1.25:
                    img = img[:, r:]
                else:
                    break
            else:
                if np.sum(img[:, -(r + 1)] == 0) > n_rows_of_result / 1.25:
                    img = img[:, :-(r + 1)]
                else:
                    break

        n_cols_of_result = img.shape[1]
        if laterality == 'R':
            img = img[:, : -int(n_cols_of_result / 15)]
        else:
            img = img[:, int(n_cols_of_result / 15):]

        return img

    def convert_to_8_bit(self, image_in):
        image_in = (image_in - image_in.min()) / (image_in.max() - image_in.min())
        return cv2.convertScaleAbs(image_in * 255)

    # def correct_inversion(self, image):
    #     max_intensity = np.max(image)
    #     corrected_image = (max_intensity - image)
    #     corrected_image[corrected_image == max_intensity] = 0
    #     return corrected_image




    def correct_inversion(self, img_ds):  
        """  
        Corrects pixel intensity inversion for a DICOM image dataset. 
        
        Parameters:  
            img_ds: A DICOM dataset that includes a pixel array and tags.  ﷼  
            
        Returns:  
            A NumPy array of corrected pixel values or the original pixel values   
            if the inversion tag does not exist, or None if an error occurs.  
        """  
        try:  
            image = img_ds.pixel_array  
            max_intensity = np.max(image)  
    
            # Check if the tag exists and its value  
            if (0x2050, 0x0020) in img_ds:  
                if img_ds[0x2050, 0x0020].value == 'INVERSE':  
                    
                    if (0x0028, 0x0107) in img_ds:  
                        pad = img_ds[0x0028, 0x0107].value
                        image = pad - image  
                        image[image == max_intensity] = 0  
    
                        return image  # Return the original or modified image 
                    else:
        
                        if (0x0028,0x0121) in img_ds:
                            if img_ds[0x0028,0x0121].value != 0:
                                
                                WL = [img_ds[0x0028, 0x1050].value][0][0]
                                WW = [img_ds[0x0028, 0x1051].value][0][0]
                                print((WL - WW))
                                image [image < (WL - WW)] =0 
                                
                                return image
                            
                if img_ds[0x2050, 0x0020].value == 'IDENTITY': 
                   if (0x0028,0x0121) in img_ds:
                       if img_ds[0x0028,0x0121].value != 0:
                                
                           WL = [img_ds[0x0028, 0x1050].value][0][0]
                           WW = [img_ds[0x0028, 0x1051].value][0][0]
                           print((WL - WW))
                           image [image < (WL - WW)] =0 
                                
            return image
                   
                        
        except AttributeError:  
            logging.error("Error: img_ds does not have pixel_array attribute.")  
            return None  
        except KeyError:  
            logging.error("Error: Unable to read DICOM tags.")  
            return None  
        except Exception as e:  
            logging.error(f"An unexpected error occurred: {e}")  
            return None
        
        
        
    # def makeBackgrondZeroInPadImage (self,img_ds):
    #     try:
    #         image = img_ds.pixel_array
    #         if (0x0028,0x0121) in img_ds:
    #             if img_ds[0x0028,0x0121].value != 0:
                    
    #                 WL = [img_ds[0x0028, 0x1050].value][0][0]
    #                 WW = [img_ds[0x0028, 0x1051].value][0][0]
    #                 print((WL - WW))
    #                 image [image < (WL - WW)] =0 
    #         return image
    #     except AttributeError:  
    #         logging.error("Error: img_ds does not have the tag attribute named Pixel Padding Range Limit    ")  
            




    def resize(self, image):
        target_width = 3028  
        target_height = 4096  
        
        try:  
            resized_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)  
            return resized_img
        except Exception as e:  
            print(e)
            return image
        

    def standardize_image(self, image_ds):
        
        image = self.correct_inversion(image_ds)
        print()
        #image = self.makeBackgrondZeroInPadImage (img_ds)
        image = self.annotation_removal(image,image_ds)
        
        image = cv2.normalize(image,dst=None, alpha=0, beta=4095, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)  
        #print(image.dtype)
        mean = np.mean(image[image > 0.0])
        std = np.std(image[image > 0.0])
        
        standardized = (image - mean) / std * self.ref_std + self.ref_mean
        
        #print(standardized.min(), standardized.max())
        
        #norm_standardized = ((standardized - standardized.min()) / (standardized.max() - standardized.min()))*255
        #standardized = standardized.astype('uint8')
        #print(norm_standardized.min(), norm_standardized.max())
        
        standardized =np.clip(standardized, 0, 4095).astype(np.uint16)
        standardized = self.resize(standardized)       
        norm_std_image = (standardized - standardized.min())/(standardized.max()- standardized.min())
        norm_std_image= (norm_std_image*4095).astype('uint16')
        
        return norm_std_image
    
    def plot_histograms(self, original_image, processed_image):  
        """Plots histograms of the original and processed images for comparison."""  
        # Create a figure with two subplots  
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
        
        # Flatten the images to compute histograms  
        original_hist, bins_original = np.histogram(original_image.flatten(), bins=(original_image.max()-original_image.min() +1)  , range=[original_image.min() +1, original_image.max()])  
        processed_hist, bins_processed = np.histogram(processed_image.flatten(), bins=4096, range=[1, 4096])  
    
        # Plot original image histogram  
        axes[0].bar(bins_original[:-1], original_hist, color='blue', alpha=0.7)  
        axes[0].set_title('Histogram of Original Image')  
        #axes[0].set_xlim([0, 256])  
        axes[0].set_xlabel('Pixel Intensity')  
        axes[0].set_ylabel('Frequency')  
    
        # Plot processed image histogram  
        axes[1].bar(bins_processed[:-1], processed_hist, color='orange', alpha=0.7)  
        axes[1].set_title('Histogram of Processed Image')  
        axes[1].set_xlim([0, 4096])  
        axes[1].set_xlabel('Pixel Intensity')  
        axes[1].set_ylabel('Frequency')  
    
        # Show the plots  
        plt.tight_layout()  
        plt.show()  
        
    def remove_skin_with_erosion(self,image, width_fraction = 10):

        kernel_size=int(image.shape[1]/width_fraction)
        kernel = np.ones((kernel_size,kernel_size), np.uint16)
        eroded_img = cv2.erode(image, kernel)
        zeros_img=np.zeros(eroded_img.shape,dtype='uint16')
        mask_of_eroded_img= np.ma.masked_where(eroded_img!=0, zeros_img , copy=True)  
        image=(mask_of_eroded_img.mask)*(image)    
    
        return image
                    

#============================ # Example usage # ===========================#


# ref_image should be a preloaded numpy array representing the reference mammography image.

ref_image_path = "/Mammo_IMGs/File_Mg_1 (25)/Nov 23 2020 (10)/PAT01/MG/MG10"
ref_image = (pydicom.dcmread(ref_image_path)).pixel_array

# plt.imshow(ref_image,'gray')
# plt.show()


preprocessor = MammographyPreprocessor(ref_image)
# print(preprocessor.ref_mean)
# print(preprocessor.ref_std)



# main_path='/Volumes/Expansion/Adata/VinDrMammo/'
# imges_path= 'F:/VinDrMammo/images'

# excel_content=glob.glob('{}/*.csv'.format(main_path))
# data=pd.read_csv(excel_content[0])

# main_path = "/Volumes/Expansion/Adata/VinDrMammo/images/"
# case_paths = glob.glob('{}*'.format(main_path))
# c=0
# inv_c = 0
# for item in case_paths[4500+220+45:]:
#     c+=1
#     #print(item)
#     MGs = glob.glob('{}*'.format(item + '/'))
#     for MgPath in MGs[:]:
        
        
#         print(MgPath)
        
#         input_image = (pydicom.dcmread(MgPath).pixel_array)

#         plt.imshow(input_image,'gray')
#         plt.show()
#         img_ds = pydicom.dcmread(MgPath)
#         processed_image = preprocessor.standardize_image(img_ds)

#         plt.imshow(processed_image ,'gray')
#         plt.show()

# main_path = "/Volumes/Expansion/ASUS/G Drive/INBreast/INbreast Release 1.0/AllDICOMs/"
# case_paths = glob.glob('{}*.dcm'.format(main_path))
# for item in case_paths[:]:
#     c+=1

#     print(item)
    
#     input_image = (pydicom.dcmread(item).pixel_array)

#     plt.imshow(input_image,'gray')
#     plt.show()
#     img_ds = pydicom.dcmread(item)
#     processed_image = preprocessor.standardize_image(img_ds)

#     plt.imshow(processed_image ,'gray')
#     #plt.title(MgPath)
#     plt.show()








