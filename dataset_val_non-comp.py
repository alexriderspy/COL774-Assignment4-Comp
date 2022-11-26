dataframe_val_x = pd.read_csv(os.path.join(directory,'non_comp_test_x.csv'))
dataframe_val_y = pd.read_csv(os.path.join(directory, 'non_comp_test_y.csv'))

input_ids = []
attention_masks = []

sentences = dataframe_val_x['Title'].values
labels = dataframe_val_y['Genre'].values

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
    b_labels = batch[2].to(device)
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
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    pred_flat = np.argmax(logits, axis=1).flatten()
    lis += pred_flat.tolist()
    labels_flat = labels.flatten()
    total_eval_accuracy += np.sum(pred_flat == labels_flat) / len(labels_flat)


# Report the final accuracy for this validation run.
avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
