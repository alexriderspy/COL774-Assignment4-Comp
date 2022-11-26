import csv

dataframe_val_x = pd.read_csv(os.path.join(directory,'comp_test_x.csv'))

input_ids = []
attention_masks = []

sentences = dataframe_val_x['Title'].values
labels = dataframe_val_x['Id'].values

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(sent,  add_special_tokens = True,  max_length = 64,  pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

dataset2 = TensorDataset(input_ids, attention_masks, labels)

test_dataloader = DataLoader(
            dataset2, # The validation samples.
            sampler = SequentialSampler(dataset2), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

lis = []
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    with torch.no_grad():        
        outputs = model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask,
                              labels=None)
        logits = outputs['logits']

    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    pred_flat = np.argmax(logits, axis=1).flatten()
    lis += pred_flat.tolist()

predicted_vals = []
iter = 0

for x in lis:
    predicted_vals.append((iter, x))
    iter += 1

header = ['Id','Genre']
directory_out = '/kaggle/working/'
with open(os.path.join(directory_out,'output.csv'), 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(header)
    # write a row to the csv file
    writer.writerows(predicted_vals)
