dataset_test_comp = ImageLoader(dataframe_x = pd.read_csv(os.path.join(directory,'comp_test_x.csv')), dataframe_y = pd.read_csv(os.path.join(directory, 'sample_submission.csv')), root_dir = os.path.join(directory, 'images/images/'), transform = transform)

predicted_vals = []
        
dataloader_test_comp = DataLoader(dataset = dataset_test_comp, batch_size = batch_size, shuffle=False, num_workers = 0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(30)]
    n_class_samples = [0 for _ in range(30)]

    iter = 0
    for images, labels in dataloader_test_comp:
        images = images.to(device, dtype=torch.float)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        for i in range(batch_size):
            
            pred = predicted[i]
            predicted_vals.append([iter,pred.item()])
            iter += 1
header = ['Id','Genre']
directory_out = '/kaggle/working/'
with open(os.path.join(directory_out,'output.csv'), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    # write a row to the csv file
    writer.writerows(predicted_vals)