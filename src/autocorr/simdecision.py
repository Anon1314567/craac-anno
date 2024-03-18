# Functions for Goal 2 - calculate similarities and make good of vectors/outputs
import os, glob, json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE

def gen_vec(image,string,OUT_DIR):
    vec_files = [os.path.basename(x) for x in sorted(glob.glob(OUT_DIR+"/"+image+'/*.json')) if os.path.basename(x).startswith(string)]
    vecs = []
    bboxes = []
    catids = []
    for vec_file in vec_files:
        with open(os.path.join(OUT_DIR,image,vec_file), 'r') as file:
            vc = json.load(file)
            vec = vc['vector']
            bbox = vc['bbox']
            bbox = [round(x,2) for x in bbox]
            catid = vc['category_id']
            vecs.append(vec)
            bboxes.append(bbox)
            catids.append(catid)
    return vecs, bboxes, catids

def sim_calc(del_vec,pred_vec):
    dlv = np.array(del_vec)
    pv = np.array(pred_vec)
    sim = dlv.dot(pv)/(np.linalg.norm(dlv)*np.linalg.norm(pv))
    return sim

def ratio_calc(bbox):
    if type(bbox[0]) == list:               # if bbox is a list of bboxes
        H = [x[3] - x[1] for x in bbox]
        W = [x[2] - x[0] for x in bbox]
    else:                                   # if only a single bbox
        H = [bbox[3] - bbox[1]]
        W = [bbox[2] - bbox[0]]
    
    ratio = np.divide(W,H)
    return ratio


def breakAccum(deleted_accum):
    del_bboxes, del_info, del_index = [], [], []
    if len(deleted_accum[0][0])==4:        # in case you have categories too
        for j in deleted_accum:                                                            
            temp_bbox = [v for i, v, im_id, cat in j]
            temp_info = [(i,im_id, cat) for i, v, im_id, cat in j]
            del_bboxes.append(temp_bbox)
            del_info.append(temp_info)
        # for i, data in enumerate(deleted_accum):
        #     for j, instance_info in enumerate(data):
        #         instance, bbox, im_id, cat = instance_info
        #         del_bboxes.append(bbox)
        #         del_info.append((instance,im_id, cat))
        #         # del_index.append((i,j))
    else:
        for j in deleted_accum:                                                              
            temp_bbox = [v for i, v, im_id in j]
            temp_info = [(i,im_id) for i, v, im_id in j]
            del_bboxes.append(temp_bbox)
            del_info.append(temp_info)
        # for i, data in enumerate(deleted_accum):
        #     for j, instance_info in enumerate(data):
        #         instance, bbox, im_id = instance_info
        #         del_bboxes.append(bbox)
        #         del_info.append((instance,im_id))
        #         # del_index.append((i,j))
    
    return del_bboxes, del_info

# ---------------------
# Developed for ICCCBE - no clustering
# ---------------------

def compare_dict(image_id, pred_bboxes, j, **kwargs):
    sim_dict = {
        "image": image_id,
        "pred_instance": j,
        "pred_bbox": [round(elem,2) for elem in pred_bboxes[j]],
        # "gt_instance": i,
        # "gt_bbox": gt_bboxes[i],
    }
    for k, v in kwargs.items():
        sim_dict[k] = v
    return sim_dict

def storeBank(image_id, deleted_vecs, del_bboxes, del_info, pred_vecs, pred_bboxes, sim_rm_thres, new_cat=False, ratio_thres=0.2, pred_cat=[]):
    store_rm_dicts = []
    for i, del_vec in enumerate(deleted_vecs):
        del_ratio = ratio_calc(del_bboxes[i])
        for j, pred_vec in enumerate(pred_vecs):
            sim = sim_calc(del_vec, pred_vec)                                           # compute similarities
            ratio = ratio_calc(pred_bboxes[j])[0]
            for k in range(sim.ndim):
                ratio_bool = (ratio <= (del_ratio[k] + ratio_thres) and ratio >= (del_ratio[k] - ratio_thres))          # check if W/H ratio of pred_bbox is similar to del_bbox
                if new_cat == False:                                                                                    # for deletion/addition
                    cat_bool = ('pred_cat' not in locals() or (len(pred_cat)>0 and pred_cat[j] == del_info[i][0][2]))
                else:                                                                                                   # for category changes
                    cat_bool = True
                if sim[k] > sim_rm_thres and ratio_bool == True and cat_bool == True:
                    if new_cat==True:                                                                                   # adjusted for category checks
                        store_dict = compare_dict(image_id, pred_bboxes,j,gt_bbox=del_bboxes[i][k], from_image=del_info[i][0][1],similarity=sim[k],new_cat=del_info[i][0][2])
                    else:
                        store_dict = compare_dict(image_id, pred_bboxes,j,gt_bbox=del_bboxes[i][k], from_image=del_info[i][0][1],similarity=sim[k])
                    store_rm_dicts.append(store_dict)
    return store_rm_dicts

# ---------------------
# Developed for operations with clustering
# ---------------------
def clusterBank(image_id, deleted_vecs, deleted_accum, pred_vecs, pred_bboxes, sim_rm_thres, ratio_thres=0.3, pred_cat=[], new_cat=False):
    store_rm_dicts = []
    del_bboxes, del_info = breakAccum(deleted_accum)
    for i, del_img in enumerate(deleted_vecs):
        del_ratio = ratio_calc(del_bboxes[i])
        for j, pred_vec in enumerate(pred_vecs):
            for k in range(len(del_img)):
                # print(f'testing instance {k} of image {i} against pred_instance {j} of image {image_id}')
                sim = sim_calc(del_img[k], pred_vec)                            # compute similarities
                ratio = ratio_calc(pred_bboxes[j])[0]                           # compute pred_bbox ratio
                # try:
                #     print(f'image {image_id} pred_instance {j} in category {pred_cat[j]} has ratio {ratio} is compared with GT image {i} instance {k} in category {del_info[i][k][2]} of ratio {del_ratio[k]}, returning a similarity {sim} ')
                # except:
                #     pass
                ratio_tol = del_ratio[k] * ratio_thres                                                                  # ratio tolerance set within ratio_thres of the del_bbox
                ratio_bool = (ratio <= (del_ratio[k] + ratio_tol) and ratio >= (del_ratio[k] - ratio_tol))              # check if W/H ratio of pred_bbox is similar to del_bbox

                if new_cat == False:                                                                                    # for deletion/addition
                    cat_bool = (not pred_cat or (len(pred_cat)>0 and pred_cat[j] == del_info[i][k][2]))
                else:                                                                                                   # for category changes
                    cat_bool = True
                
                if sim > sim_rm_thres and ratio_bool == True and cat_bool == True:                                   # insert other filter criteria here                
                    if new_cat==True:                                                                                   # adjusted for category checks
                        store_dict = clusterDict(image_id, pred_bboxes[j], j, del_info[i][k][1], del_info[i][k][0], del_bboxes[i][k], sim, new_cat=del_info[i][k][2])
                    else:
                        store_dict = clusterDict(image_id, pred_bboxes[j], j, del_info[i][k][1], del_info[i][k][0], del_bboxes[i][k], sim)
                    store_rm_dicts.append(store_dict)
    return store_rm_dicts                                                   

def clusterDict(image_id, pred_bbox, j, i, k, gt_bbox, sim, **kwargs):
    sim_dict = {
        "pred_image": image_id,
        "pred_bbox": [round(elem,2) for elem in pred_bbox],
        "pred_instance": j,
        "gt_image": i,
        "gt_instance": k,
        "gt_bbox": gt_bbox,
        "similarity": sim
    }
    for k, v in kwargs.items():
        sim_dict[k] = v
    return sim_dict

def clusterIdxLoop(sim_del, deleted_accum, j):
    want_img = [x['gt_image'] for x in sim_del if x['pred_instance'] == j]
    want_instance = [x['gt_instance'] for x in sim_del if x['pred_instance'] == j]
    gt_img_idx, gt_instance_idx = [], []
    # print(want_img, want_instance)
    for s, data in enumerate(deleted_accum):
        for x, instance_info in enumerate(data):
            instance, _, image_id, _ = instance_info  # Access the image_id from deleted_accum
            # print(f"Checking gt_image_id: {image_id}, gt_instance: {instance}")
            for y, want in enumerate(want_img):
                if image_id == want and instance == want_instance[y]:
                    gt_img_idx.append(s)
                    gt_instance_idx.append(x)
    return gt_img_idx, gt_instance_idx

# Create a cluster for every pred_instance
def flat_vec(vec_list):
    flat_vecs = []
    for k, dlv in enumerate(vec_list):
        norm_vec = np.array(dlv)/np.linalg.norm(dlv)
        flat_vecs.append(norm_vec)
    return flat_vecs

# Create a two dimensional plot to illustrate the outcome via t-SNE projection of the embeddings
def tsne_plot(train_vecs, test_vec, k, image_id, image_instance, labels):
    tsne = TSNE(2, perplexity=k, verbose=1)
    train_vecs = train_vecs + [test_vec]
    # print(type(train_vecs), len(train_vecs))
    tsne_proj = tsne.fit_transform(np.array(train_vecs))
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = mpl.colormaps['tab20']
    clabel = ['deletion', 'addition', 'category change']
    fig, ax = plt.subplots(figsize=(8,8))
    for lab in range(k):
        label_bool = [x for x, val in enumerate(labels) if val == lab+1]
        if not label_bool:
            continue
        else:
            ind_start = label_bool[0]
            ind_end = label_bool[-1]
        ax.scatter(tsne_proj[ind_start:ind_end,0],tsne_proj[ind_start:ind_end,1], c=np.array(cmap(lab)).reshape(1,4), label = clabel[lab] ,alpha=0.5)
    ax.scatter(tsne_proj[-1,0],tsne_proj[-1,1], c="red", label = "test" ,alpha=0.5)     # draw the test_vec point
    ax.legend(fontsize='large', markerscale=2)
    plt.title(f'image {image_id} instance {image_instance} in t-SNE space')
    plt.show()

def clusterAction(store_rm_dicts, store_add_dicts, store_cat_dicts, decision, dict_add):
    if decision == 1:
        store_rm_dicts.append(dict_add)
        return store_rm_dicts
    elif decision == 2:
        store_add_dicts.append(dict_add)
        return store_add_dicts
    elif decision == 3:
        store_cat_dicts.append(dict_add)
        return store_cat_dicts
    else:
        pass


def clusterDecision(image_id, 
                    deleted_vecs, deleted_accum, 
                    added_vecs, added_accum,
                    chcats_vecs, chcats_accum,
                    pred_vecs, pred_bboxes, 
                    sim_thres, ratio_thres,
                    **kwargs):
    store_rm_dicts, store_add_dicts, store_cat_dicts = [], [], []
    pred_catid = kwargs.get('pred_catid', [])
    # Step 1 - find similar instances
    sim_del, sim_add, sim_chcats = [], [], []               # just in case if they are not created
    if deleted_accum: 
        sim_del = clusterBank(image_id, deleted_vecs, deleted_accum, pred_vecs, pred_bboxes, sim_thres[0], ratio_thres[0], pred_cat=pred_catid)
    if added_accum:
        sim_add = clusterBank(image_id, added_vecs, added_accum, pred_vecs, pred_bboxes, sim_thres[1], ratio_thres[1], pred_cat=[])
    if chcats_accum:
        sim_chcats = clusterBank(image_id, chcats_vecs, chcats_accum, pred_vecs, pred_bboxes, sim_thres[2], ratio_thres[2], pred_cat=pred_catid, new_cat=True)
        # print("sim_chcats:", sim_chcats)
    
    # Step 2 - making decision (via clustering or others)
    for j, pred_vec in enumerate(pred_vecs):                # perhaps to parse broken deleted_accum to iterate by every pred_instance of every image?
        # test_vec = pred_vecs[j]
        if sim_del: 
            gt_img_idx, gt_instance_idx = clusterIdxLoop(sim_del, deleted_accum, j)
            del_cluster = [deleted_vecs[pos][gt_instance_idx[s]] for s, pos in enumerate(gt_img_idx)]
            # del_cluster = [deleted_vecs[x['gt_image']][x['gt_instance']] for x in sim_del if x['pred_instance'] == j]
            flat_deleted_vecs = flat_vec(del_cluster)
            # print("similar deletions:", len(del_cluster))
        else:
            flat_deleted_vecs = []
        
        if sim_add: 
            gt_img_idx, gt_instance_idx = clusterIdxLoop(sim_add, added_accum, j)
            add_cluster = [added_vecs[pos][gt_instance_idx[s]] for s, pos in enumerate(gt_img_idx)]
            # add_cluster = [added_vecs[x['gt_image']][x['gt_instance']] for x in sim_add if x['pred_instance'] == j]
            flat_added_vecs = flat_vec(add_cluster)
            # print("similar additions:", len(add_cluster))
        else:
            flat_added_vecs = []

        if sim_chcats: 
            gt_img_idx, gt_instance_idx = clusterIdxLoop(sim_chcats, chcats_accum, j)
            chcats_cluster = [chcats_vecs[pos][gt_instance_idx[s]] for s, pos in enumerate(gt_img_idx)]
            flat_chcats_vecs = flat_vec(chcats_cluster)
            # new_cat = [x['new_cat'] for x in sim_chcats if x['pred_instance'] == j]
            matching_items = [x for x in sim_chcats if x['pred_instance'] == j]
            if matching_items:
                max_item = max(matching_items, key=lambda x: x['similarity'])
                new_cat = max_item['new_cat']
            else:
                new_cat = []
            # print("new_cat:", new_cat)
            # print("similar category changes:", len(chcats_cluster)) 
        else:
            flat_chcats_vecs = []
        
        train_vecs = flat_deleted_vecs + flat_added_vecs + flat_chcats_vecs
        labels = [1] * len(flat_deleted_vecs) + [2] * len(flat_added_vecs) + [3] * len(flat_chcats_vecs)

        if len(train_vecs) == 0:                    # no similar samples
            action = None
            print(f'image {image_id} pred_instance {j} can skip the similarity test')
        elif len(train_vecs) == 1:                  # take the recommended action
            print("pred_instance:", j, "takes action ", labels[0])
            action = labels[0]
        elif len(train_vecs) == 2:                  # find the datapoint with similarity and not bother with kNN
            sim = sim_calc(train_vecs, pred_vec)
            action = labels[sim.tolist().index(max(sim))]
            print("pred_instance:", j, "takes action ", action)
        else:
            # Create and fit a k-NN classifier with a specific k
            k = 3 
            clf = Pipeline(steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
            clf.fit(train_vecs, labels)
            action = clf.predict(np.array(pred_vec).reshape(1, -1))[0]
            print("pred_instance:", j, "takes action ", action)
            # tsne_plot(train_vecs, pred_vec, k, image_id, j, labels) # Step 4 - output with t-SNE
        
        # confirm what to add
        if action==3:
            dict_add = compare_dict(image_id, pred_bboxes, j, new_cat=new_cat)
        else:
            dict_add = compare_dict(image_id, pred_bboxes, j)

        # Step 3 - parse into store_x_dicts
        if action: 
            clusterAction(store_rm_dicts, store_add_dicts, store_cat_dicts, action, dict_add)
    
    # pred_instance can be deleted or cat changed. proposal_instance can be added
    if pred_catid:
        return store_rm_dicts, store_cat_dicts
    else:
        return store_add_dicts, None