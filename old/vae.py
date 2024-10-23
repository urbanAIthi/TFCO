def vae_loss(recon_x, x, mu, logvar):
    reconstruction_loss = nn.BCELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence
# Training loop
def train_vae(vae, dataloader, device):
    # Define the optimization algorithm
    optimizer = optim.Adam(vae.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Training loop
    num_epochs = 10000
    vae = vae.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            reconstructed_batch, mu, logvar = vae(batch)

            # Compute loss
            loss = vae_loss(reconstructed_batch, batch, mu, logvar)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Perform scheduler step after each epoch
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}, Learning Rate: {current_lr:.6f}")
        if epoch % 10 == 0:
            trained_out = get_current_output(autoencoder, dataset, 0)
            show_image(trained_out, show=True)