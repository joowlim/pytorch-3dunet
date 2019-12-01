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
    if targetSize[2] < z :
        dif_half = int((z-targetSize[2])/2)
        for i in range(z_data.shape[2]):
            z_data[:,:,i] = data[:,:,dif_half+i]
    else:
        dif_half= int((targetSize[2]-z)/2)
        #print('dif_half:',dif_half)
        j = 0

        #print('zdata:',z_data.shape)
        #print('range:',dif_half,z_data.shape[2]-dif_half)
        for i in range(dif_half,z_data.shape[2]-dif_half-1):
            z_data[:,:,i] = data[:,:,j]
            j = j+1

    #for d in range(z_data.shape[2]):
    #    vis = np.zeros([z_data.shape[0], z_data.shape[1], 3])
    #    vis[:, :, 0] = z_data[:, :, d]
    #    vis[:, :, 1] = z_data[:, :, d]
    #    vis[:, :, 2] = z_data[:, :, d]
    #    cv2.imshow('img data', vis)
    #    cv2.waitKey(0)

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
    return new_data

def main():
    DataRootPath = 'CS470/nrrd_data/MRI'
    H5RootPath = 'CS470/h5_data/MRI'
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
            ctdata, header = nrrd.read(data_path)
            original_size = ctdata.shape
            #print('original size',original_size)
            vmin = np.amin(ctdata)

            ctdata = ctdata - vmin
            vmax = np.amax(ctdata)
            normalized_ctdata = np.divide(ctdata,vmax)
            #print('LOADING DONE : CTDATA')

            seg_pmma, header = nrrd.read(seg_path)
            seg_verte, header = nrrd.read(verte_path)
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

            pad_ctdata = resize_with_padding(final_ctdata,[128,128,64])
            pad_pmma = resize_with_padding(final_pmma,[128,128,64])
            pad_vert = resize_with_padding(final_vert, [128,128,64])

            pad_ctdata = np.transpose(pad_ctdata,(2,0,1))
            pad_pmma = np.transpose(pad_pmma, (2, 0, 1))
            pad_vert = np.transpose(pad_vert, (2, 0, 1))
            #print('new ct dim:',pad_ctdata.shape)
            #exit(0)

            pad_pmma = np.reshape(pad_pmma,(1,pad_pmma.shape[0],pad_pmma.shape[1],pad_pmma.shape[2]))
            pad_vert = np.reshape(pad_vert, (1, pad_vert.shape[0], pad_vert.shape[1], pad_vert.shape[2]))
            #print('new pad pmma:',pad_pmma.shape)
            #exit(0)

            pad_label = np.concatenate((pad_pmma, pad_vert),axis=0)
            #print('pad_label size:',pad_label.shape)
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

                for h in range(vis_target.shape[1]):
                    for w in range(vis_target.shape[2]):
                        if pad_label[0,d,h,w] == 1:
                            vis[h,w,:] = vis[h,w,:]*0.5 + (0,0,0.5)
                        if pad_label[1,d,h,w] == 1:
                            vis[h, w, :] = vis[h, w, :] * 0.5 + (0, 0.5, 0.0)
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

            f = h5py.File(os.path.join(H5RootPath,pre+'.hdf5'),'w')
            resolution = f.create_dataset("resolution", data=pad_ctdata.shape)
            minmax = f.create_dataset("minmax", data=[vmin,vmax])
            data = f.create_dataset("raw", data=pad_ctdata)
            #pmma_seg = f.create_dataset("pmma", data=pad_pmma.astype(np.int64))
            #verte_seg = f.create_dataset("verte", data=pad_vert.astype(np.int64))
            label = f.create_dataset("label",data=pad_label.astype(np.int64))
            f.close()
            print('SIZE IS CHANGED FROM',original_size,' TO ',pad_ctdata.shape)
            print('WRITE %s IS DONE'%(os.path.join(H5RootPath,pre+'.hdf5')))


        else:
            print('FILE:', data_path)
            print('FILE:', seg_path)
            print('FILE:', verte_path)
            print("NO FILE")
            exit()
    exit(0)

if __name__ == '__main__':
    main()
