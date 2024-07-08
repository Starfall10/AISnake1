import wandb
import random
for i in range(1):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project-3",

        # track hyperparameters and run metadata
        config={

        }
    )

    # simulate training
    epochs = 10
    offset = random.randint(0,10)

    acc = [0,1,2,3,4,5,6,7]
    loss = [9,8,7,6,5,4,3,2]
    

        # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()