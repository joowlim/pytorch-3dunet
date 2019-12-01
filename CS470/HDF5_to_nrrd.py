import numpy as np
import nrrd
import h5py
import glob, os
import cv2
import json

def decideLabels(prediction_map, threshold = 0.7):
    verte_label = np.zeros([prediction_map.shape[1], prediction_map.shape[2], prediction_map.shape[3]])
    pmma_label = np.zeros([prediction_map.shape[1], prediction_map.shape[2], prediction_map.shape[3]])

    for d in range(prediction_map.shape[1]):
        for h in range(prediction_map.shape[2]):
            for w in range(prediction_map.shape[3]):
                pmap = prediction_map[:,d,h,w]
                maxidx = np.argmax(pmap)
                if pmap[maxidx]>threshold:
                    if maxidx == 1 or maxidx == 3:
                        pmma_label[d,h,w]=1
                    if maxidx == 2 or maxidx == 3:
                        verte_label[d,h,w]=1
                #exit(0)
    return verte_label, pmma_label

def recover_raw_data(mri_data,prediction_data, prediction_channel, raw_data_size, depth_range):
    print(raw_data_size[0], raw_data_size[1], raw_data_size[2])

    data_without_depth_padding = mri_data[depth_range[0]:depth_range[1], :, :]
    prediction_without_depth_padding = prediction_data[:,depth_range[0]:depth_range[1], :, :]

    longer_length = 0
    if raw_data_size[1] > raw_data_size[2]:
        longer_length = raw_data_size[1]
    else:
        longer_length = raw_data_size[2]

    ##<< [ SC : Resized with hw padding]
    data_resized_back = np.zeros([raw_data_size[0], longer_length, longer_length])
    prediction_resized_back = np.zeros([prediction_channel,raw_data_size[0], longer_length, longer_length])

    for d in range(raw_data_size[0]):
        data_resized_back[d, :, :] = cv2.resize(data_without_depth_padding[d, :, :], (longer_length, longer_length),
                                                interpolation=cv2.INTER_LINEAR)
        for c in range(prediction_channel):
            prediction_resized_back[c,d, :, :] = cv2.resize(prediction_without_depth_padding[c,d, :, :], (longer_length, longer_length),
                                                interpolation=cv2.INTER_LINEAR)

    #print('data_resized_back:', data_resized_back.shape)
    ##<< [ SC : Eliminate hw padding]
    # dif_hw_half = 0
    if raw_data_size[1] > raw_data_size[2]:
        dif_hw_half = int((raw_data_size[1] - raw_data_size[2]) / 2)
        #print(dif_hw_half)
        data_without_padding = data_resized_back[:, :, dif_hw_half:(longer_length - dif_hw_half)]
        prediction_without_padding = prediction_resized_back[:,:, :, dif_hw_half:(longer_length - dif_hw_half)]
        #segVerte_without_padding = segVerte_resized_back[:, :, dif_hw_half:(longer_length - dif_hw_half)]
        #segPmma_without_padding = segPmma_resized_back[:, :, dif_hw_half:(longer_length - dif_hw_half)]

    else:
        dif_hw_half = int((raw_data_size[2] - raw_data_size[1]) / 2)
        #print(dif_hw_half)
        data_without_padding = data_resized_back[:, dif_hw_half:(longer_length - dif_hw_half), :]
        prediction_without_padding = prediction_resized_back[:,:, dif_hw_half:(longer_length - dif_hw_half), :]
        #segVerte_without_padding = segVerte_resized_back[:, dif_hw_half:(longer_length - dif_hw_half), :]
        #segPmma_without_padding = segPmma_resized_back[:, dif_hw_half:(longer_length - dif_hw_half), :]

    final_raw_data = np.zeros([raw_data_size[0], raw_data_size[1], raw_data_size[2]])
    #final_verte_data = np.zeros([raw_data_size[2], raw_data_size[0], raw_data_size[1]])
    #final_pmma_data = np.zeros([raw_data_size[2], raw_data_size[0], raw_data_size[1]])
    final_prediction_data = np.zeros([prediction_channel,raw_data_size[0], raw_data_size[1], raw_data_size[2]])
    for d in range(raw_data_size[0]):
        final_raw_data[d, :, :] = cv2.resize(data_without_padding[d, :, :], (raw_data_size[2], raw_data_size[1]))
        for c in range(prediction_channel):
            final_prediction_data[c,d,:,:] = cv2.resize(prediction_without_padding[c,d, :, :], (raw_data_size[2], raw_data_size[1]))

    rotated_raw_data = np.zeros([final_raw_data.shape[0],final_raw_data.shape[2],final_raw_data.shape[1]])
    rotate_prediction_data = np.zeros([prediction_channel, final_prediction_data.shape[1],final_prediction_data.shape[3],final_prediction_data.shape[2]])

    for d in range(raw_data_size[0]):
        rotated_raw_data[d,:,:] = cv2.rotate(final_raw_data[d,:,:],cv2.ROTATE_90_CLOCKWISE)
        for c in range(prediction_channel):
            rotate_prediction_data[c,d,:,:] = cv2.rotate(final_prediction_data[c,d,:,:],cv2.ROTATE_90_CLOCKWISE)

    return rotated_raw_data, rotate_prediction_data

def main():
    SourceRootPath = 'CS470/h5_data_1130/test'
    ResultRootPath = 'CS470/result/191130_p64_cdatav2'
    HeaderRootPath = 'CS470/h5_data_1130/header'
    TargetRootPath = os.path.join(ResultRootPath,'nrrd')
    AllFileList = os.listdir(ResultRootPath)
    print(AllFileList)
    targetList = []

    for file in AllFileList:
        if 'predictions' in file:
            tmp = os.path.splitext(file)[0]
            tmp = tmp.split('_predictions')[0]
            targetList.append(tmp)
    targetList.sort()
    for pre in targetList:
        if os.path.exists(os.path.join(TargetRootPath,pre+"_mri.nrrd")):
            print('ALREADY EXISTS')
            continue

        #if pre == targetList[0]:
        #    continue
        path_mri_header = os.path.join(HeaderRootPath,pre+"_mri_header.json")
        path_verte_header = os.path.join(HeaderRootPath,pre+"_verte_header.json")
        path_pmma_header = os.path.join(HeaderRootPath,pre+"_pmma_header.json")
        path_prediction = os.path.join(ResultRootPath,pre+"_predictions.h5")
        path_source = os.path.join(SourceRootPath,pre+".hdf5")

        if os.path.exists(path_prediction) and os.path.exists(path_mri_header) and os.path.exists(path_pmma_header) and os.path.exists(path_verte_header):
            data_path = os.path.join(TargetRootPath, pre + "_mri.nrrd")
            pmma_path = os.path.join(TargetRootPath, pre + "_PMMA.nrrd")
            verte_path = os.path.join(TargetRootPath, pre + "_Vertebra.nrrd")

            print(path_verte_header)
            with open(path_mri_header) as json_file:
                mri_header = json.load(json_file)
            with open(path_verte_header) as json_file:
                verte_header = json.load(json_file)
            with open(path_pmma_header) as json_file:
                pmma_header = json.load(json_file)

            ##<< [ SC : Resize mask tensor to raw resolution and eliminate padded region]
            source_obj = h5py.File(path_source, "r")
            result_obj = h5py.File(path_prediction, "r")
            prediction = result_obj['/predictions'] # (4x64x128x128)
            raw_data = source_obj['/raw'] # (64x128x128)
            raw_data_size = source_obj['/raw_resolution']
            depth_roi = source_obj['/depth_range']
            value_minmax = source_obj['/minmax']
            #print(value_minmax[0], value_minmax[1])
            #exit(0)

            final_raw_data, final_predictions = recover_raw_data(raw_data,prediction,4,raw_data_size,depth_roi)


            ##<< [ SC : Visualize Results]
            vis_data = np.zeros([final_raw_data.shape[1],final_raw_data.shape[2],3])
            # for t in range(final_raw_data.shape[0]):
            #     for h in range(final_raw_data.shape[1]):
            #         for w in range(final_raw_data.shape[2]):
            #             vis_data[h,w,0]=final_raw_data[t,h,w]
            #             vis_data[h, w, 1] = final_raw_data[t, h, w]
            #             vis_data[h, w, 2] = final_raw_data[t, h, w]
            #
            #             vis_data[h,w,0] += 0.5*final_predictions[1,t,h,w]
            #             #vis_data[h, w, 1] += 0.5 * final_predictions[3, t, h, w]
            #     cv2.imshow('test',vis_data)
            #     cv2 .waitKey(0)

            ##<< [ SC : Decide final label of predction result. Three mask tensor expected. ]
            verte_mask, pmma_mask = decideLabels(final_predictions)

            # vis_data = np.zeros([final_raw_data.shape[1], final_raw_data.shape[2], 3])
            # for t in range(final_raw_data.shape[0]):
            #     for h in range(final_raw_data.shape[1]):
            #         for w in range(final_raw_data.shape[2]):
            #             vis_data[h, w, 0] = final_raw_data[t, h, w]
            #             vis_data[h, w, 1] = final_raw_data[t, h, w]
            #             vis_data[h, w, 2] = final_raw_data[t, h, w]
            #
            #             if verte_mask[t,h,w] == 1:
            #                 vis_data[h, w, 0] += 0.5
            #             if pmma_mask[t,h,w] == 1:
            #                 vis_data[h,w,1] += 0.5
            #
            #
            #
            #     cv2.imshow('test', vis_data)
            #     cv2.waitKey(0)

            ##<< [ SC : Write nrrd file ]
            final_raw_data = np.transpose(final_raw_data,(1,2,0))
            final_raw_data = final_raw_data*value_minmax[1]
            verte_mask = np.transpose(verte_mask, (1, 2, 0))
            pmma_mask = np.transpose(pmma_mask, (1, 2, 0))

            nrrd.write(data_path,final_raw_data,mri_header)
            nrrd.write(verte_path, verte_mask, verte_header)
            nrrd.write(pmma_path, pmma_mask, pmma_header)


        else:
            print('FILE:', data_path)
            print('FILE:', pmma_path)
            print('FILE:', verte_path)
            print("NO FILE")
            exit()
    exit(0)

if __name__ == '__main__':
    main()
