class DefaultConfig():
    #input 
    image_scale = [768,1024]
    dataset_dir = "/workspace/mnt/cache/MaskData/VOC2007/"

    #backbone
    backbone_choice= "vovnet39" #"vovnet39"or"resnet18"
    pretrained=False
    freeze_stage_1=False
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=2
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=False

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    BATCH_SIZE=6
    EPOCHS=30
    WARMPUP_STEPS_RATIO=0.12
    GLOBAL_STEPS=1
    LR_INIT=5e-5
    LR_END=1e-6

    #inference
    score_threshold=0.2
    nms_iou_threshold=0.2
    max_detection_boxes_num=150
    inference_dir = "/workspace/mnt/group/algorithm/anchao/backbone/FCOS_Mask/images/"

    #resume
    resume_path = None