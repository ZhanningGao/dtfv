#./IDT/release/DenseTrackStab /tmp/v_Archery_g01_c01.avi | gzip > v_Archery_g01_c01.gz

./IDT/release/DenseTrackStab /tmp/v_Archery_g01_c01.avi | ./compute_fv_gpu ../data/ucfTEST.pca.lst ../data/ucfTEST.codebook.lst ./ucf