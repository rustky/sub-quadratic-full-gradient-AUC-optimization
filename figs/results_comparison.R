# Load the necessary libraries
library(ggplot2)
library(plyr)
library(data.table)

# set your working directory to where you downloaded the experiment cvs's
setwd('~/Documents/monsoon') 
# Each key needs to be the folder name of the experiments
experiment.desc <- list("experiment4_3_1"= "LIBAUC + PESG",
                        "experiment2_3_1"= "Squared Hinge + SGD",
                        "experiment5_1"="Logistic Loss + SGD")
                        # "experiment4_2"= "LIBAUC + PESG")
experiment.list  <- c("experiment4_3_1", "experiment2_3_1", "experiment5_1")
# Set up empty lists
for(seed.to.show in c(1,5,10,15,20)){
  for(set.to.show in c("CIFAR10","STL10", "CAT_VS_DOG")){
    for(ratio.to.show in c(0.001,0.01,0.1,0.5)){
result.dt.list <- list()
plot.frame.list <- list()
# For loop of over experiments 
for(experiment in experiment.list){
  results.csv.glob <- paste(experiment,"*.csv",sep='/')
  (result.csv.vec <- Sys.glob(results.csv.glob))
  wc.cmd <- paste("wc -l", results.csv.glob)
  nline.dt <- fread(cmd=wc.cmd, col.names=c("lines", "file"))
  # Only grab CSV's that have more than one line written to them
  some.csv <- nline.dt[lines>1][file!="total", file]
  lines.idxs <- which(nline.dt[,'lines'] > 1)
  lines <- unlist(nline.dt[lines.idxs, 'lines'], use.names = FALSE) - 1
  split_csv <- strsplit(some.csv, '-|=')
  split.table <- data.table(do.call(rbind, split_csv))
  set.col <- split.table[,'V12']
  set.vec <- rep(unlist(set.col[,"V12"]),times = lines[-length(lines)])
  split.seed.col <- strsplit(unlist(split.table[,"V18"]),'[.]')
  seed.table <- do.call(rbind,split.seed.col)
  seed.vec <- rep(unlist(seed.table[,1]),times =lines[-length(lines)])
  result.dt <- data.table(result.csv=some.csv)[, {
    fread(result.csv)
  }, by=result.csv]
  result.dt$dataset = set.vec
  result.dt$seed = seed.vec
  result.dt$experiment = experiment.desc[[paste(experiment)]]
  result.dt.list[[paste(experiment)]] <- result.dt

  # Control which dataset is shown in figure
  # set.to.show = 'CIFAR10'
  # Control which imratio is shown in figure
  # ratio.to.show = 0.5
  # seed.to.show = 1
  bool.vec <- which((result.dt[,'dataset'] == set.to.show) &
                      (result.dt[,'imratio'] == ratio.to.show)&
                      (result.dt[,'seed'] == seed.to.show))
  plot.frame.list[[paste(experiment)]] <- result.dt[bool.vec,]
}
plot.dt <- data.table(do.call(rbind.fill, plot.frame.list))
# Create batch size facetting variable
plot.dt[,'batch' :=  factor(plot.dt[['batch_size']],
                            c(sort(c(10,50,100,500,1000,5000)),
                              'full'))]
# Find where the max test AUC is achieved for each experiment
max.idxs <- plot.dt[, .I[which.max(test_auc)],by=experiment]
max.dt <- plot.dt[max.idxs$V1,]

# Create big grid of test AUC's, annotated with max AUC locations
plot <- ggplot(plot.dt) +
  geom_line(data = plot.dt,
            aes(epoch,test_auc, color= experiment))+
  facet_grid(lr ~ batch,
             labeller = label_both,
             scales='free')+
  geom_point(data = max.dt,
             aes(epoch, test_auc,fill=experiment),
             color='black',
             shape = 21,
             size =.9)+
  ggtitle(paste(set.to.show,ratio.to.show,seed.to.show, sep=" "))
  ggsave(paste(set.to.show,ratio.to.show,seed.to.show, sep="-"),
         plot=plot,
         device="png",
         width=9.02,
         height = 5.74,
         units='in')
}}}
# Create tables for experiments that were trained using the LIBAUC loss versus
# ours
libauc.bool <- which(plot.dt[,'experiment'] == 'LIBAUC + SGD' | 
                       plot.dt[,'experiment'] == 'LIBAUC + PESG')
ours.bool <- which(plot.dt[,'experiment'] == 'Squared Hinge + LBFGS' | 
                     plot.dt[,'experiment'] == 'Squared Hinge + SGD')

libauc.dt <- plot.dt[libauc.bool,]
ours.dt <- plot.dt[ours.bool,]

# Create big grid of train our square hinge losses 
ggplot() +
  geom_line(data=ours.dt,
            aes(epoch,train_our_square_hinge, color=experiment))+
  facet_grid(lr ~ batch_size)+
  ggtitle(paste(set.to.show,ratio.to.show, sep=" "))

# Create big grid of train LIBAUC losses
ggplot()+
  geom_line(data=libauc.dt,
            aes(epoch,test_libauc_loss, color=experiment))+
  facet_grid(lr ~ batch_size)+
  scale_y_log10() +
  ggtitle(paste(set.to.show,ratio.to.show,"log10 y-axis", sep=" "))

