import numpy as np
import nrrd
import h5py
import os, sys
sys.path.append(os.getcwd())
import cv2
import torch
from unet3d.metrics import DiceCoefficient, MeanIoU

import scipy.ndimage.filters as filter

def applygaussian3D(input_3d, in_sigma, in_truncate):
    #print('input 3d:',input_3d.shape)
    return filter.gaussian_filter(input_3d, sigma=in_sigma, truncate=in_truncate)

def interpolation(input_3d, win_size):
    assert win_size%2 == 1

    hw = int(win_size/2)

    output_3d = input_3d.copy()

    for d in range(hw,input_3d.shape[0]-hw):
        for h in range(hw,input_3d.shape[1]-hw):
            for w in range(hw,input_3d.shape[2]-hw):
                value = 0.0
                for d2 in range(d-hw, d+hw+1):
                    for h2 in range(h-hw,h+hw+1):
                        for w2 in range(w-hw, w+hw+1):
                            value += input_3d[d2,h2,w2]

                output_3d[d,h,w] = value/np.power(win_size,3)
    return output_3d


def make_video(raw, fp, prediction_path, output_path):
    # making video
    #loaded_pred = np.array(fp['/predictions'])
    #loaded_raw = np.array(fi['/raw'])
    loaded_pred = fp
    loaded_raw = raw

    ##<< [ SC : Test Gaussian 3D ]
    # s=2
    # w=5
    # t = (((w-1)/2)-0.5)/s
    # shapenum = loaded_pred.shape[0]
    # for i in range(shapenum):
    #     loaded_pred[i] = applygaussian3D(loaded_pred[i],s,t)
    ##<< [ SC : Test interpolation 3D ]
    #for i in range(loaded_pred.shape[0]):
    #     loaded_pred[i] = interpolation(loaded_pred[i],3)
    seg_pmma_out = loaded_pred[1,:,:,:]
    seg_pmma = loaded_pred[2,:,:,:]
    seg_vert = loaded_pred[3,:,:,:]

    ##<< [ SC : Modify

    vis_result = np.zeros([loaded_raw.shape[1], loaded_raw.shape[2],3])
    pmma_result = np.zeros([loaded_raw.shape[1], loaded_raw.shape[2],3])
    vert_result = np.zeros([loaded_raw.shape[1], loaded_raw.shape[2], 3])
    pmma_out_result = np.zeros([loaded_raw.shape[1], loaded_raw.shape[2], 3])

    video_saved = []
    # for d in range(loaded_raw.shape[2]):
    #     for h in range(loaded_raw.shape[0]):
    #         for w in range(loaded_raw.shape[1]):
    #             vis_result[h,w,0] = loaded_raw[h,w,d]
    #             vis_result[h, w, 1] = loaded_raw[h, w,d]
    #             vis_result[h, w, 2] = loaded_raw[h, w,d]
    #     print('depth:', d,'/',loaded_raw.shape[2])
    #     cv2.imshow('raw_data',vis_result)
    #     cv2.waitKey(0)
    #print('loadedraw shape', loaded_raw.shape)
    for d in range(loaded_raw.shape[0]):
        vis_result[:,:,0] = loaded_raw[d,:,:]
        vis_result[:, :, 1] = loaded_raw[d,:, :]
        vis_result[:, :, 2] = loaded_raw[d,:, :]

        pmma_result[:,:,0] = seg_pmma[d,:,:]
        pmma_result[:, :, 1] = seg_pmma[d,:, :]
        pmma_result[:, :, 2] = seg_pmma[d,:, :]

        vert_result[:, :, 0] = seg_vert[d, :, :]
        vert_result[:, :, 1] = seg_vert[d, :, :]
        vert_result[:, :, 2] = seg_vert[d, :, :]

        pmma_out_result[:,:,0] = seg_pmma_out[d,:,:]
        pmma_out_result[:,:,1] = seg_pmma_out[d,:,:]
        pmma_out_result[:,:,2] = seg_pmma_out[d,:,:]

        #print('depth:', d,'/',loaded_raw.shape[0])
        #cv2.imshow('raw_data',vis_result)
        #cv2.imshow('seg_data',seg_result)

        vis_horizontal = np.hstack((vis_result,pmma_out_result, pmma_result,vert_result))
        video_saved.append(vis_horizontal)

    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_file_name = os.path.splitext(os.path.basename(prediction_path))[0] + ".avi"
    output_file_path = os.path.join(output_path, output_file_name)
    print('WRITING:', output_file_path)
    vw = cv2.VideoWriter(output_file_path,fourcc,5.0,(128*4,128))

    for i in range(len(video_saved)):
        gray = cv2.normalize(video_saved[i],None,255,0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vw.write(gray)
    vw.release()  

def calc_accuracy(fi, fp):
    # calculate accuracy
    #label_pred = np.array(fp['/predictions']) # [4, 64, 128, 128]
    label_pred=fp
    # s = 2
    # w = 5
    # t = (((w - 1) / 2) - 0.5) / s
    # for i in range(label_pred.shape[0]):
    #     label_pred[i] = applygaussian3D(label_pred[i],s,t)
    #for i in range(label_pred.shape[0]):
    #    label_pred[i] = interpolation(label_pred[i], 3)

    label_pred_ext = torch.tensor([label_pred]) # [1, 4, 64, 128, 128]
    #label_gt = fi['/label'] # [64, 128, 128]
    label_gt = fi
    label_gt_ext = torch.tensor([label_gt]) # [1, 64, 128, 128]



    # print ("label_pred : ", label_pred_ext.shape)
    # print ("label_gt : ", label_gt_ext.shape)
    
    # iou
    iou = MeanIoU()(label_pred_ext, label_gt_ext)
    # dice
    dice = DiceCoefficient()(label_pred_ext, label_gt_ext)
    return iou.item(), dice.item()

def main(input_path, result_path):
    filenames = os.listdir(input_path)
    input_files = []
    for i in range(len(filenames)):
        input_files.append(os.path.join(input_path, filenames[i]))

    filenames = os.listdir(result_path)
    result_files = []
    for i in range(len(filenames)):
        result_files.append(os.path.join(result_path, filenames[i]))

    assert len(input_files) == len(result_files)

    output_path = result_path + "_vis"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    input_files.sort()
    result_files.sort()
    ious = []
    dices = []
    print('np power',np.power(3,3))
    for i in range(len(input_files)):
        prediction_path = result_files[i]
        input_path = input_files[i]
        if os.path.exists(prediction_path) and os.path.exists(input_path):
            fi = h5py.File(input_path, "r")
            fp = h5py.File(prediction_path, "r")

            fi_array = np.array(fi['/label'])
            fp_array = np.array(fp['/predictions'])
            raw_array = np.array(fi['/raw'])

            #for i in range(fp_array.shape[0]):
            #    fp_array[i] = interpolation(fp_array[i], 5)
            
            make_video(raw_array, fp_array, prediction_path, output_path)

            iou, dice = calc_accuracy(fi_array, fp_array)
            ious.append(iou)
            dices.append(dice)
        else:
            print("NO FILE")
            exit()
    print ("Average iou : ", sum(ious)/len(ious))
    print ("Average dice : ", sum(dices)/len(dices))
    
    f = open(os.path.join(output_path, "accuracy.txt"), "w")
    f.write("Average iou : %f\n" % (sum(ious)/len(ious)))
    f.write("Average iou : %f\n" % (sum(dices)/len(dices)))
    f.close()

if __name__ == '__main__':
    input_path = 'CS470/h5_data_1130/test'
    result_path = 'CS470/result/191201_final'
    main(input_path, result_path)
