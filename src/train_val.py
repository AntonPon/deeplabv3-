import torch

def train(model, data_loader, epoch, optimizer, loss_fucntion,  metric, board_writer=None, scheduler=None, device='cpu'):
    train_loss = 0.
    train_miou = 0.
    for idx, input_batch in enumerate(data_loader):
        img_batch = input_batch['imgs'].to(device)
        masks_batch = input_batch['masks'].to(device)

        output_masks = model(img_batch)
        loss = loss_fucntion(masks_batch, output_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_miou += metric(masks_batch, output_masks).item()
    if board_writer is not None:
        board_writer.add_scalar('train/loss', train_loss / len(data_loader), epoch)
        board_writer.add_scalar('train/miou', train_miou / len(data_loader), epoch)


def val(model, loss_func, metric, data_loader, epoch, board_writer, device='cpu'):
    with torch.no_grad():
        val_loss = 0.
        val_miou = 0.
        for idx, input_batch in enumerate(data_loader):
            img_batch = input_batch['imgs'].to(device)
            masks_batch = input_batch['masks'].to(device)

            output_masks = model(img_batch)

            loss = loss_func(masks_batch, output_masks).item()
            val_loss += loss
            miou = metric(masks_batch, output_masks).item()
            val_miou += miou
        if board_writer is not None:
            board_writer.add_scalar('val/loss', val_loss/len(data_loader), epoch)
            board_writer.add_scalar('val/miou', val_miou/len(data_loader, epoch))


