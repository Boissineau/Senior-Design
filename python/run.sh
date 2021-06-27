# python demo.py \
# --feature-type 'HoG' \
# --query-index 0

echo "Creating mat file"
python create_mat_file.py 

echo "Generating features"
python deep/generate_features.py \
--edge-image-file '../data/my_features/edge_image_file.mat' \
--model-name 'deep/network.pth' \
--batch-size 64 \
--cuda-id 0 \
--save-file '../data/my_features/feature_file.mat'

echo "Stitching together edge maps and features"
python stitch_feat_edge.py 

echo "Running the demo"
python demo_uot.py \
--feature-type 'deep' \
--algo 2



# pytorch -> create_mat_file.py -> deep/generate_features.sh -> stitch_feat_edge.py -> demo



#feature-type 'deep' uses pretrained, not using deep will use our HoG that we made in data

#python demo.py --feature-type 'deep' --query-index 0