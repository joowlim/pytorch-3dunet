import numpy as np
import nrrd
import h5py
import glob, os
import cv2

def resize_with_padding(data, targetSize):
    #print(data.ndim, len(targetSize))
    #assert data.ndim != len(targetSize), "Dimension of data is wrong in resize_with_padding()"
    if data.ndim != len(targetSize):
        print("Dimension of data is wrong in resize_with_padding()")

    h,w,z = data.shape

    z_data = np.zeros([data.shape[0],data.shape[1],targetSize[2]])

    start_depth, end_depth = 0, 0

    print('original data depth:', z)
    if targetSize[2] < z : ##<< [ SC : if original data is larger than target ]
        dif_half = int((z-targetSize[2])/2)
        start_depth = dif_half
        for i in range(z_data.shape[2]):
            z_data[:,:,i] = data[:,:,dif_half+i]
            end_depth = dif_half+i
    else:
        dif_half= int((targetSize[2]-z)/2)
        j = 0
        start_depth = dif_half
        end_depth = dif_half+z
        for i in range(dif_half,dif_half+z):
            z_data[:,:,i] = data[:,:,j]
            j = j+1

    print('roi output:', end_depth - start_depth)
    #for d in range(z_data.shape[2]):
    #    vis = np.zeros([z_data.shape[0], z_data.shape[1], 3])
    #    vis[:, :, 0] = z_data[:, :, d]
    #    vis[:, :, 1] = z_data[:, :, d]
    #    vis[:, :, 2] = z_data[:, :, d]
    #    cv2.imshow('img data', vis)
    #    cv2.waitKey(0)

    ##<< [ SC : Padding according to height and width ]
    new_data = np.zeros([targetSize[0],targetSize[1],targetSize[2]])

    if h > w:
        dif_half = int(h-w/2)
        npad = ((0,0),(dif_half,dif_half))

    elif h < w:
        dif_half = int((w-h)/2)
        npad = ((dif_half,dif_half),(0,0))
    else:
        npad=((0,0),(0,0))

    for i in range(new_data.shape[2]):
        #print('zdata shape:',z_data.shape)
        tmp = np.pad(z_data[:,:,i],npad,'constant',constant_values=(0))
        new_data[:,:,i] = cv2.resize(tmp,(targetSize[0],targetSize[1]),interpolation=cv2.INTER_LINEAR)

    #print(new_data)
    #print(np.amax(new_data))
    return new_data, [start_depth, end_depth]

def main():
    DataRootPath = 'CS470/MRI_crop_nrrd'
    H5RootPath = 'CS470/tmp'
    AllFileList = os.listdir(DataRootPath)
    targetList = []

    for file in AllFileList:
        if 'cropped' in file:
            tmp = os.path.splitext(file)[0]
            tmp = tmp.split('_cropped')[0]
            targetList.append(tmp)

    for pre in targetList:
        if os.path.exists(os.path.join(H5RootPath,pre+".hdf5")):
            print('ALREADY EXISTS')
            continue
        data_path = os.path.join(DataRootPath,pre+"_cropped.nrrd")
        seg_path = os.path.join(DataRootPath,pre+"_PMMA.nrrd")
        verte_path = os.path.join(DataRootPath,pre+"_Vertebra.nrrd")

        #print(data_path, seg_path, verte_path)
        #exit(0)

        if os.path.exists(data_path) and os.path.exists(seg_path) and os.path.exists(verte_path):
            ctdata, header_mri = nrrd.read(data_path)
            #print(type(header))
            #print('length:',len(header))
            #print(header.popitem(last=False), type(header.popitem(last=False)))

            #print(header[0], type(header))
            #exit(0)
            original_size = ctdata.shape
            #print('original size',original_size)
            vmin = np.amin(ctdata)

            ctdata = ctdata - vmin
            vmax = np.amax(ctdata)
            normalized_ctdata = np.divide(ctdata,vmax)
            #print('LOADING DONE : CTDATA')

            seg_pmma, header_pmma = nrrd.read(seg_path)
            seg_verte, header_verte = nrrd.read(verte_path)
            #print('LOADING DONE : SEG INFO')

            if not(seg_pmma.shape == seg_verte.shape == ctdata.shape):
                print('SHAPE OF DATA IS NOT IDENTICAL!',ctdata.shape, seg_pmma.shape, seg_verte.shape)
                print('FILE:',data_path)
                print('FILE:', seg_path)
                print('FILE:', verte_path)
                exit(1)

            #normalized_ctdata = cv2.rotate(normalized_ctdata,cv2.ROTATE_90_COUNTERCLOCKWISE)
            final_ctdata = np.zeros([normalized_ctdata.shape[1], normalized_ctdata.shape[0],normalized_ctdata.shape[2]])
            final_pmma = np.zeros([seg_pmma.shape[1], seg_pmma.shape[0],seg_pmma.shape[2]])
            final_vert = np.zeros([seg_pmma.shape[1], seg_pmma.shape[0], seg_pmma.shape[2]])

            for d in range(normalized_ctdata.shape[2]):
                final_ctdata[:,:,d] = cv2.rotate(normalized_ctdata[:,:,d],cv2.ROTATE_90_COUNTERCLOCKWISE)
                final_pmma[:,:,d] = cv2.rotate(seg_pmma[:,:,d],cv2.ROTATE_90_COUNTERCLOCKWISE)
                final_vert[:,:,d] = cv2.rotate(seg_verte[:,:,d],cv2.ROTATE_90_COUNTERCLOCKWISE)

            pad_ctdata, roi_ctdata= resize_with_padding(final_ctdata,[128,128,64])
            pad_pmma, roi_pmma = resize_with_padding(final_pmma,[128,128,64])
            pad_vert, roi_vert = resize_with_padding(final_vert, [128,128,64])

            #print(roi_ctdata)
            #exit(0)

            pad_ctdata = np.transpose(pad_ctdata,(2,0,1))
            pad_pmma = np.transpose(pad_pmma, (2, 0, 1))
            pad_vert = np.transpose(pad_vert, (2, 0, 1))*2
            #print('new ct dim:',pad_ctdata.shape)
            #exit(0)

            #pad_pmma = np.reshape(pad_pmma,(1,pad_pmma.shape[0],pad_pmma.shape[1],pad_pmma.shape[2]))
            #pad_vert = np.reshape(pad_vert, (1, pad_vert.shape[0], pad_vert.shape[1], pad_vert.shape[2]))*2
            #print('new pad pmma:',pad_pmma.shape)
            #exit(0)

            pad_label = pad_pmma+pad_vert
            #print('pad_label size:',pad_label.shape, 'min,max:',np.amax(pad_label),np.amin(pad_label))
            #exit(0)
            #print(pad_ctdata.shape)

            ##<< [SC : Visualization ]
            vis_target = pad_ctdata
            print('vistarget:',vis_target.shape)

            for d in range(vis_target.shape[0]):
                vis = np.zeros([vis_target.shape[1],vis_target.shape[2],3])
                vis[:,:,0] = vis_target[d,:,:]
                vis[:,:,1] = vis_target[d,:,:]
                vis[:,:,2] = vis_target[d,:,:]
                #print('final_pmma size:',final_pmma.shape)

                # for h in range(vis_target.shape[1]):
                #     for w in range(vis_target.shape[2]):
                #         if pad_label[d,h,w] == 1:
                #             #print('1')
                #             vis[h,w,:] = vis[h,w,:]*0.5 + (0,0,0.5)
                #         elif pad_label[d,h,w] == 2:
                #             vis[h, w, :] = vis[h, w, :] * 0.5 + (0, 0.5, 0.0)
                #         elif pad_label[d,h,w] == 3:
                #              vis[h, w, :] = vis[h, w, :] * 0.5 + (0.5, 0.0, 0.0)
                #print('idx:',d,'/',vis_target.shape[2])
                #cv2.imshow('img data', vis)
                #cv2.waitKey(0)

            # for d in range(vis_target.shape[2]):
            #     vis = np.zeros([vis_target.shape[0],vis_target.shape[1],3])
            #     vis[:,:,0] = vis_target[:,:,d]
            #     vis[:,:,1] = vis_target[:,:,d]
            #     vis[:,:,2] = vis_target[:,:,d]
            #     #print('final_pmma size:',final_pmma.shape)
            #
            #     for h in range(vis_target.shape[0]):
            #         for w in range(vis_target.shape[1]):
            #             if pad_pmma[h,w,d] == 1:
            #                 vis[h,w,:] = vis[h,w,:]*0.5 + (0,0,0.5)
            #             if pad_vert[h,w,d] == 1:
            #                 vis[h, w, :] = vis[h, w, :] * 0.5 + (0, 0.5, 0.0)
            #     #print('idx:',d,'/',vis_target.shape[2])
            #     #cv2.imshow('img data', vis)
            #     #cv2.waitKey(0)

            ##<< [ SC : Writing h5 files ]

            #print(final_ctdata.shape)
            WRITING_H5 = True
            if WRITING_H5:
                raw_resolution_tmp = np.array([final_ctdata.shape[2],final_ctdata.shape[0],final_ctdata.shape[1]])
                f = h5py.File(os.path.join(H5RootPath,pre+'.hdf5'),'w')
                resolution = f.create_dataset("raw_resolution", data=raw_resolution_tmp)
                depth_roi = f.create_dataset("depth_range", data=roi_ctdata)
                minmax = f.create_dataset("minmax", data=[vmin,vmax])
                data = f.create_dataset("raw", data=pad_ctdata)
                label = f.create_dataset("label",data=pad_label.astype(np.int64))
                f.close()

            WRITING_JSON = True
            if WRITING_JSON :
                import json
                for key, value in header_pmma.items():
                    if isinstance(value,np.ndarray):
                        header_pmma[key]= value.tolist()

                for key, value in header_verte.items():
                    if isinstance(value,np.ndarray):
                        header_verte[key]= value.tolist()

                for key, value in header_mri.items():
                    if isinstance(value,np.ndarray):
                        header_mri[key]= value.tolist()

                with open(os.path.join(H5RootPath,pre+'_pmma_header.json'),'w') as f:
                    json.dump(header_pmma, f)
                with open(os.path.join(H5RootPath,pre+'_verte_header.json'),'w') as f:
                    json.dump(header_verte, f)
                with open(os.path.join(H5RootPath,pre+'_mri_header.json'),'w') as f:
                    json.dump(header_mri, f)

            print('SIZE IS CHANGED FROM',original_size,' TO ',pad_ctdata.shape)
            print('WRITE %s IS DONE'%(os.path.join(H5RootPath,pre+'.hdf5')))
            #exit(0)

        else:
            print('FILE:', data_path)
            print('FILE:', seg_path)
            print('FILE:', verte_path)
            print("NO FILE")
            exit()
    #exit(0)

if __name__ == '__main__':
    main()
