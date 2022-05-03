# Load necessary libraries
library(data.table)
library(ggplot2)

# Hand picked files from grid display based on how each algorithm performs in
# the first 100 epochs or so
LIBAUC.PESG <- "experiment4_3_1/pretrained=False-loss_name=libauc_loss-lr=0.1-batch_size=100-imratio=0.01-dataset=CIFAR10-model=ResNet20-optimizer=PESG-seed=20.csv"
square.hinge.SGD <- "experiment2_3_1/pretrained=False-loss_name=square_hinge_test-lr=0.01-batch_size=10-imratio=0.01-dataset=CIFAR10-model=ResNet20-optimizer=SGD-seed=20.csv"

# Assign experiment description to each file name
experiment.desc <- list(
                        "Squared Hinge + SGD" = square.hinge.SGD,
                        "LIBAUC + PESG" = LIBAUC.PESG)
filenames <- c(LIBAUC.PESG, square.hinge.SGD)
result_list <- {}
# Loop over experiments
for(files in seq_along(experiment.desc)){
  temp.dt  <- data.table(results.csv = experiment.desc[[files]])[, {
    fread(results.csv)
  }, by=results.csv]
  temp.dt[, 'experiment' := names(experiment.desc)[[files]]]
  # Create a new column with the time difference between the completion of the
  # first epoch and every subsequent epoch in minutes
  epoch_one_time <- temp.dt[[1,'wall_time']]
  temp.dt[, 'time_from_epoch_one' := 
            as.numeric(difftime(wall_time, epoch_one_time,unit='mins'))]
  result_list[[paste(files)]] <- temp.dt
}
plot.dt <- do.call(rbind, result_list)

# Plot test auc vs epochs
ggplot(data = plot.dt)+
  geom_line(aes(
    x = epoch,
    y = test_auc,
    color = experiment))+
  ggtitle("CIFAR10 0.01")

# Plot test auc vs epochs
ggplot(data = plot.dt)+
  geom_line(aes(
    x = epoch,
    y = train_auc,
    color = experiment))+
  ggtitle("STL10 0.001")

# Create table that only includes observations that occured less that 175 mins
# from the completion of the first epoch
time.idx <- which(plot.dt[,'time_from_epoch_one'] < 175)
time.dt <- plot.dt[time.idx,]

# Plot test AUC vs time
ggplot(data= time.dt)+
  geom_line(aes(
    x = time_from_epoch_one,
    y = test_auc,
    color = experiment
  ))+
  xlab('Time from the completion of epoch one (mins)')+
  scale_y_continuous(breaks = c(0.40,0.45, 0.50, 0.55))+
  ggtitle("CIFAR10 0.01")

# Plot train and test AUC vs time for square hinge + SGD
# TODO: Add column for color differential instead of hard coding in strings to
# seperate them. This applies for all subsequent figures
ggplot(data= time.dt[which(experiment == "Squared Hinge + SGD"),])+
  geom_line(aes(
    x = time_from_epoch_one,
    y = train_our_square_hinge,
    color = "Train Our Square Hinge"
  ))+
  geom_line(aes(
    x = time_from_epoch_one,
    y = test_our_square_hinge,
    color = "Test Our Square Hinge"
  ))+
  xlab('Time from the completion of epoch one (mins)')+
  ggtitle("STL10 0.001 Squared Hinge + SGD")

# Plot train and test AUC vs time for square hinge + LBFGS
ggplot(data= time.dt[which(experiment == "Squared Hinge + LBFGS"),])+
  geom_line(aes(
    x = time_from_epoch_one,
    y = train_our_square_hinge,
    color = "Train Our Square Hinge"
  ))+
  geom_line(aes(
    x = time_from_epoch_one,
    y = test_our_square_hinge,
    color = "Test Our Square Hinge"
  ))+
  xlab('Time from the completion of epoch one (mins)')+
  ggtitle("STL10 0.001 Squared Hinge + LBFGS")

# Plot train and test AUC vs time for LIBAUC + PESG
ggplot(data= time.dt[which(experiment == "LIBAUC + PESG"),])+
  geom_line(aes(
    x = time_from_epoch_one,
    y = train_libauc_loss,
    color = "Train LIBAUC Loss"
  ))+
  geom_line(aes(
    x = time_from_epoch_one,
    y = test_libauc_loss,
    color = "Test LIBAUC Loss"
  ))+
  xlab('Time from the completion of epoch one (mins)')+
  ggtitle("STL10 0.001 LIBAUC + PESG")
