Paper implementation of [Cross Image Pixel Contrast for Semantic Segmentation (ICCV 21')](https://arxiv.org/pdf/2101.11939v4)

Code can be integrated in your semantic segmentation training pipeline using the following steps

1. Initialise ContrastiveLearner class for training and validation respectively

    ```
    contrastive_learner_train = ContrastiveLearner(region_memory_bank_size = training_set_size,
                                                     pixel_memory_bank_size = 10 * training_set_size,
                                                     num_classes = num_classes)
   
    contrastive_learner_val = ContrastiveLearner(region_memory_bank_size = validation_set_size,
                                                 pixel_memory_bank_size = 10 * validation_set_size,
                                                 num_classes = num_classes)
    ```

2. Build memory bank inside training loop (after forward pass on a batch is done)

    `contrastive_learner_train.build_memory_bank(features = features.detach(),index = index, label = y.detach())`

3. Calculate contrastive loss inside training loop

   `contrastive_loss = contrastive_learner_train.compute_loss(preds,features,y)`
   
