with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(30)]
    n_class_samples = [0 for _ in range(30)]

    for images, labels in dataloader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        
    acc = 100.0 * (n_correct/n_samples)
    print(f'Accuracy of network: {acc} %')
    
    for i in range(30):
        acc = 100.0 * (n_class_correct[i]/n_class_samples[i])
        print(f'Accuracy of classes[i]: {acc} %')