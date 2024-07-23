import torch
import torch.nn as nn

class ContrastiveLearner:
    def __init__(self,region_memory_bank_size,pixel_memory_bank_size,num_classes,device,embedding_dim,pixel_sampling_per_batch,num_anchors,k_positive,
                 k_negative, temperature):
        self.device = device
        self.temperature = temperature
        self.num_classes = num_classes
        self.pixels_to_sample_per_batch = pixel_sampling_per_batch
        self.region_memory_bank_size = region_memory_bank_size
        self.pixel_memory_bank_size = pixel_memory_bank_size
        self.num_anchors = num_anchors
        self.k_positive = k_positive
        self.k_negative = k_negative
        self.index = 0
        
        self.pixel_level_memory_bank = torch.empty((self.num_classes,0,embedding_dim)).to(dtype = torch.float32, device = device)
        self.region_level_memory_bank = torch.zeros((self.num_classes,region_memory_bank_size,embedding_dim)).to(dtype = torch.float32,device = device)
    
    def build_memory_bank(self,features,label,index):
        self.build_region_level_bank(features,label,index)
        self.build_pixel_level_bank(features,label)

    def compute_loss(self,predictions : torch.Tensor, features : torch.Tensor, labels = torch.Tensor):
        loss = 0.0

        for class_id in range(self.num_classes):
            anchors = self.sample_anchors(predictions = predictions, features = features, labels = labels,class_id = class_id)
            loss += self.compute_anchor_loss(anchors,class_id)
            del anchors 
            torch.cuda.empty_cache()

        return loss
    
    def compute_anchor_loss(self, anchors : torch.Tensor, class_id : int):
        loss = 0.0
        num_anchors = anchors.shape[0]
        for index in range(num_anchors):
            anchor = anchors[index]
            positive_examples,negative_examples = self.sample_examples(anchor = anchor, class_id = class_id)
            loss += self.loss_function(positive_examples,negative_examples,anchor)
            del anchor,positive_examples,negative_examples
            torch.cuda.empty_cache()

        return loss
    
    def loss_function(self, positive_examples : torch.Tensor, negative_examples : torch.Tensor, anchor : torch.Tensor):
        # examples K,D
        # anchor    ,D

        positive_examples_dot_product =  torch.sum(positive_examples * anchor, dim = 1) / self.temperature              # K
        positive_examples_exp = torch.exp(positive_examples_dot_product)                                                # K

        negative_examples_dot_product = torch.sum(anchor * negative_examples, dim = 1) / self.temperature
        negative_examples_exp = torch.exp(negative_examples_dot_product)
        negative_examples_exp_sum = torch.sum(negative_examples_exp)

        contrastive_loss_per_example = -torch.log( positive_examples_exp / (positive_examples_exp + negative_examples_exp_sum))
        contrastive_loss = torch.mean(contrastive_loss_per_example)

        del positive_examples_dot_product,positive_examples_exp,positive_examples,negative_examples,anchor,negative_examples_dot_product
        del negative_examples_exp,negative_examples_exp_sum,contrastive_loss_per_example
        torch.cuda.empty_cache()

        return contrastive_loss

    def sample_examples(self, anchor :torch.Tensor, class_id : int):
        # memory bank C,T,D
        # anchors     D

        memory_bank = torch.cat([self.pixel_level_memory_bank,self.region_level_memory_bank],dim = 1)
        negative_examples = torch.cat([memory_bank[ : class_id, :, :],memory_bank[class_id + 1 : , :, :]],dim = 0)       # K,D
        positive_examples = memory_bank[class_id , : , :]

        dot_product_negative_samples = torch.sum(negative_examples * anchor, dim = 1)
        dot_product_positive_samples = torch.sum(positive_examples * anchor, dim = 1)

        hardest_negative_sample_indices = torch.argsort(dot_product_negative_samples,descending=True)[ : int(0.1*dot_product_negative_samples.shape[0])]
        hardest_positive_sample_indices = torch.argsort(dot_product_positive_samples)[ : int(0.1*dot_product_positive_samples.shape[0])]

        hardest_negative_samples = negative_examples[hardest_negative_sample_indices, :]
        hardest_positive_samples = positive_examples[hardest_positive_sample_indices, :]


        shuffling_indices_positive = torch.randperm(hardest_positive_samples.shape[0])
        shuffling_indices_negative = torch.randperm(hardest_negative_samples.shape[0])

        semi_hard_positive_examples = hardest_positive_samples[shuffling_indices_positive , :]
        semi_hard_negative_examples = hardest_negative_samples[shuffling_indices_negative , :]

        semi_hard_positive_examples = semi_hard_positive_examples[ : self.k_positive, :]
        semi_hard_negative_examples = semi_hard_negative_examples[ : self.k_negative, :]

        del negative_examples,positive_examples,dot_product_negative_samples,dot_product_positive_samples,hardest_negative_sample_indices,hardest_positive_sample_indices
        del shuffling_indices_negative,shuffling_indices_positive
        torch.cuda.empty_cache()

        return semi_hard_positive_examples, semi_hard_negative_examples
     
    def sample_anchors(self,predictions : torch.Tensor,features : torch.Tensor, labels : torch.Tensor, class_id : int):
        # preds -    B,H,W
        # labels -   B,H,W
        # features - B,D,H,W

        num_segmentation_hard_anchors = self.num_anchors // 2
        
        flattened_preds = torch.flatten(predictions, start_dim = 0)                          # B*H*W
        flattened_labels = torch.flatten(labels, start_dim = 0)                              # B*H*W
        features = torch.flatten(torch.permute(features,dims = (1,0,2,3)),start_dim = 1)     # D,B*H*W

        incorrectly_classified_pixels = torch.sum(torch.where(((flattened_labels!=flattened_preds) & (flattened_labels == class_id)),1,0)).item()
        num_segmentation_hard_anchors = min(num_segmentation_hard_anchors,incorrectly_classified_pixels)
        num_randomly_sampled_anchors = self.num_anchors - num_segmentation_hard_anchors

        shuffling_index = torch.randperm(features.shape[1])
        shuffled_features = features[:, shuffling_index]                                               # D,K
        shuffled_features = torch.permute(features,dims =  (1,0))                                      # K,D
        randomly_sampled_anchors = shuffled_features[ : num_randomly_sampled_anchors, :]

        incorrectly_classified_pixel_mask = ((flattened_labels != flattened_preds) & (flattened_labels == class_id))
        incorrectly_classified_pixel_features = features[: , incorrectly_classified_pixel_mask]                                              # D,K 
        shuffling_index = torch.randperm(incorrectly_classified_pixel_features.shape[1])
        segmentation_hard_pixel_features_shuffled = incorrectly_classified_pixel_features[ : , shuffling_index]
        segmentation_hard_pixel_features_shuffled = torch.permute(segmentation_hard_pixel_features_shuffled,dims = (1,0))                    # K,D
        segmentation_hard_pixel_features_random = segmentation_hard_pixel_features_shuffled[ : num_segmentation_hard_anchors, :]

        anchors = torch.cat([randomly_sampled_anchors,segmentation_hard_pixel_features_random],dim = 0)

        del predictions,features,labels,flattened_labels,flattened_preds,incorrectly_classified_pixel_features,incorrectly_classified_pixel_mask
        del incorrectly_classified_pixels,shuffled_features,shuffling_index,randomly_sampled_anchors,segmentation_hard_pixel_features_random
        del segmentation_hard_pixel_features_shuffled
        torch.cuda.empty_cache()

        return anchors  

    def build_pixel_level_bank(self,features : torch.Tensor, label : torch.Tensor):
        
        pixels_to_sample = self.pixels_to_sample_per_batch
        flattened_features = torch.permute(features, dims = (1,0,2,3))           # D,B,H,W
        flattened_features = torch.flatten(flattened_features,start_dim = 1)     # D,B*H*W
        flattened_label = torch.flatten(label,start_dim = 0)                     # B*H*W

        sampled_pixel_features = []
        for class_id in range(self.num_classes):
            num_pixels = torch.sum(torch.where(flattened_label == class_id,1,0)).item()
            pixels_to_sample = min(num_pixels,pixels_to_sample)

        for class_id in range(self.num_classes):
            class_features = flattened_features[:,(flattened_label == class_id)]         # D,K
            class_features = torch.permute(class_features,dims = (1,0))                  # K,D

            num_pixels = class_features.shape[0]
            random_indices = torch.randperm(num_pixels)

            class_features_shuffled = class_features[random_indices, :]                 
            class_features_shuffled = torch.unsqueeze(class_features_shuffled,dim = 0)   # 1,K,D
            class_features_shuffled = class_features_shuffled[:, : pixels_to_sample, :]

            sampled_pixel_features.append(class_features_shuffled)

        batch_pixel_features = torch.cat(sampled_pixel_features,dim = 0)
        self.update_pixel_level_memory_bank(batch_pixel_features)

        del flattened_features,flattened_label,class_features,random_indices,class_features_shuffled,sampled_pixel_features,batch_pixel_features,features,label
            
        torch.cuda.empty_cache()

    def update_pixel_level_memory_bank(self,batch_pixel_features : torch.Tensor):
        self.pixel_level_memory_bank = torch.cat([self.pixel_level_memory_bank,batch_pixel_features],dim = 1)
        if self.pixel_level_memory_bank.shape[1] > self.pixel_memory_bank_size:
            self.pixel_level_memory_bank = self.pixel_level_memory_bank[:, (-self.pixel_memory_bank_size) : ,:]

    def build_region_level_bank(self,features : torch.Tensor ,label : torch.Tensor) -> None:

        feature_class_wise_bank = []
        features = features.permute((1,0,2,3))                                     # D,B,H,W
        label = torch.squeeze(label, dim = 1)
        for class_id in range(self.num_classes):
            class_mask = torch.where(label == class_id,1.0,0.0)                    # B,H,W
            feature_sum = torch.sum(features * class_mask, dim = (-1,-2))          # D,B
            class_pixels_per_image = torch.sum(class_mask,dim = (-1,-2))           # B
            feature_average  = feature_sum / class_pixels_per_image                # D,B
            feature_average = feature_average.permute((1,0))                       # B,D
            feature_average = torch.unsqueeze(feature_average,dim = 0)             # 1,B,D
            feature_class_wise_bank.append(feature_average)


        region_feature_vectors = torch.cat(feature_class_wise_bank,dim = 0)        # C,B,D
        self.update_region_level_memory_bank(region_feature_vectors)

        del feature_class_wise_bank,features,label,class_mask,feature_sum,class_pixels_per_image,feature_average,region_feature_vectors
        torch.cuda.empty_cache()

    def update_region_level_memory_bank(self,region_feature_vectors : torch.Tensor):
        batch_size = region_feature_vectors.shape[1]
        self.region_level_memory_bank[:,batch_size * (self.index) : min(self.region_memory_bank_size,batch_size * (self.index + 1)), :] = region_feature_vectors
        self.index = self.index + 1

        if batch_size * (self.index+1) >= self.region_level_memory_bank.shape[1]:
            self.index = 0



        

        
            
