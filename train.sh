
## To reproduce "Deep All" for DG
python train_DG.py --source photo cartoon sketch --target art_painting
python train_DG.py --source art_painting cartoon sketch --target photo
python train_DG.py --source photo art_painting sketch --target cartoon
python train_DG.py --source photo cartoon art_painting --target sketch


## To reproduce "Deep All" for DA
python train_DA.py --source photo cartoon sketch --target art_painting
python train_DA.py --source art_painting cartoon sketch --target photo
python train_DA.py --source photo art_painting sketch --target cartoon
python train_DA.py --source photo cartoon art_painting --target sketch