# Load the necessary libraries
library(ggplot2)
library(plyr)
library(data.table)
library(stringr)

# set your working directory to where you downloaded the experiment cvs's
setwd('~/Documents/monsoon') 
# Each key needs to be the folder name of the experiments
# 
# "experiment5_1_1"="Logistic Loss + SGD"
experiment.desc <- list("experiment6_1_1" = "Square Hinge + LBFGS",
                        "experiment7_1_1" = "Logistic Loss + LBFGS",
                        "experiment8_1_1" = "LIBAUC + PESG")
experiment.list  <- c("experiment6_1_1", "experiment7_1_1", "experiment8_1_1")
# experiment.desc <- list("experiment4_3_1"= "LIBAUC + PESG",
#                         "experiment2_3_1"= "Squared Hinge + SGD",
#                         "experiment5_1_1"= "Logistic Loss + SGD")
# experiment.list  <- c("experiment4_3_1", "experiment2_3_1", "experiment5_1_1")
# Set up empty lists
result.dt.list <- list()
plot.frame.list <- list()
max.validation.list <- list()
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
  set.names = str_extract(some.csv, "(?<=dataset=).*(?=-model)")
  set.vec <- rep(set.names,times = lines[-length(lines)])
  seed.names = str_extract(some.csv, "(?<=seed=).*(?=.csv)")
  seed.vec <- rep(seed.names,times =lines[-length(lines)])
  result.dt <- data.table(result.csv=some.csv)[, {
    fread(result.csv)
  }, by=result.csv]
  result.dt$dataset = set.vec
  result.dt$seed = seed.vec
  result.dt$experiment = experiment.desc[[paste(experiment)]]
  result.dt.list[[paste(experiment)]] <- result.dt
}
plot.dt <- data.table(do.call(rbind.fill, result.dt.list))
max.by.seed.dt <- plot.dt[,.SD[which.max(validation_auc), .(batch_size,lr)],
                    by=list(experiment,imratio,dataset,seed)]
best.param.dt <- plot.dt[max.by.seed.dt, on=list(batch_size, lr, experiment, imratio, dataset,seed)]
tall.dt <- nc::capture_melt_single(best.param.dt, set = "subtrain|validation|test", "_", metric="auc|our_square_hinge|libauc_loss")
tall.dt[,seed.numeric := as.integer(seed)]
wide.dt <- data.table::dcast(tall.dt,epoch+experiment+set+dataset+metric+imratio+seed.numeric~.,list(median,max,min),value.var = "value")

ggplot(wide.dt[(metric=='auc')&(set=='validation')&(imratio!=0.5)])+
  geom_line(aes(epoch,value_median,color=experiment))+
  facet_grid(dataset+imratio~seed.numeric,scales='free')+
  ylab('Validation AUC')

ggplot(wide.dt[(metric=='auc')&(set=='validation')&(imratio==0.1)&(dataset=='CIFAR10')])+
  geom_line(aes(epoch, value_median,color=experiment))+
  ylab('Validation AUC')+
  facet_grid(seed.numeric~., scales='free')

ggplot(wide.dt[(metric=='our_square_hinge')&(set!='test')&(experiment=='Square Hinge + LBFGS')])+
  geom_line(aes(epoch,value_median,color=set))+
  facet_grid(dataset+imratio~seed.numeric,scales='free')+
  ylab('Square Hinge Loss')+
  ggtitle('Subtrain Validation Loss for Square Hinge + LBFGS')

ggplot(wide.dt[(metric=='libauc_loss')&(set!='test')&(experiment=='LIBAUC + PESG')])+
  geom_line(aes(epoch,value_median,color=set))+
  facet_grid(dataset+imratio~seed.numeric,scales='free')+
  ylab('AUCM Loss')+
  ggtitle('Subtrain Validation Loss for LIBAUC + PESG')

band.dt <- data.table::dcast(tall.dt,epoch+experiment+set+dataset+metric+imratio~.,list(median,max,min),value.var = "value")
ggplot(band.dt[(dataset=='CAT_VS_DOG')&(metric=='auc')&(set=='validation')&(imratio!=0.5)])+
  geom_line(aes(epoch,value_median,color=experiment))+
  geom_ribbon(aes(epoch,ymin=value_min,ymax=value_max,fill=experiment),
              alpha=0.5)+
  ggtitle("CAT&DOG Validation AUC")+
  facet_grid(.~imratio)

bands <- ggplot(band.dt[(metric=='auc')&(imratio==0.01)&(set=='validation')])+
  geom_line(aes(epoch,value_median,color=experiment))+
  geom_ribbon(aes(epoch,ymin=value_min,ymax=value_max,fill=experiment),
              alpha=0.5)+
  ggtitle("Imbalance Ratio = 0.01 Validation AUC")+
  facet_grid(.~dataset)
print(bands)

max.validation.dt <- plot.dt[,.SD[which.max(validation_auc), .(batch_size,lr,epoch)],
                          by=list(experiment,imratio,dataset,seed)]
best.valid.dt <- plot.dt[max.validation.dt, on=list(batch_size, lr, experiment, imratio, dataset,seed, epoch)]
point.dt <- best.valid.dt[, .(test.mean = mean(test_auc), test.sd = sd(test_auc)), by=list(experiment, dataset, imratio)]
# point.dt <- best.valid.dt[, .(test.median = median(test_auc), test.25 = quantile(test_auc,0.25), test.75 = quantile(test_auc, 0.75)), by=list(experiment, dataset, imratio)]
imratio.to.show <- 0.001
points <- ggplot(point.dt[(imratio==imratio.to.show)])+
  geom_point(aes(test.mean, experiment))+
  geom_segment(aes(test.mean - test.sd,
                   experiment,
                   xend=test.mean + test.sd,
                   yend=experiment))+
  facet_grid(.~dataset, scales ='free')+
  xlab("Test AUC")+
  ylab("Loss + Algorithm")+
  ggtitle(paste("Imratio =", imratio.to.show,sep=" "))
print(points)

ggplot(max.by.seed.dt[(imratio!=0.5)])+
  geom_point(aes(batch_size, experiment))+
  facet_grid(imratio ~ dataset)+
  scale_x_log10()+
  ggtitle("Batch Size Selected For Best Validation AUC")

time.dt <- plot.dt[(epoch==0)|(epoch==199)]
diff.dt <- time.dt[,.(elapsed.time = (max(wall_time) - min(wall_time))/60), by=list(experiment, batch_size, imratio, lr, dataset)]
mean.df.dt <- diff.dt[,mean(elapsed.time),by=list(experiment, batch_size, imratio, dataset)]

ggplot(diff.dt[(elapsed.time!=0)&(experiment!='Logistic Loss + LBFGS')])+
  geom_point(aes(elapsed.time, experiment))+
  facet_grid(dataset~imratio)+
  xlab('Time (min)')+
  scale_x_continuous()+
  ggtitle('Elapsed Time Between Epoch 0 and 199')

ggplot(plot.dt[(dataset=='STL10')&(experiment=='Square Hinge + LBFGS')])+
  geom_line(aes(epoch, subtrain_our_square_hinge))+
  facet_grid(lr ~ imratio, scales='free')

# Code to subset the LIBAUC Data for a certain time limit
epoch0 <- plot.dt[epoch == 0]
elapsed.dt <- plot.dt[epoch0, on=.(experiment, seed, imratio, dataset, batch_size,lr), elapsed.time := difftime(x.wall_time,i.wall_time,units='mins'), by=.EACHI]
quantile(elapsed.dt$elapsed.time)
time.budget <- 273.4667
subset.dt <- elapsed.dt[!((elapsed.time > time.budget))]
max.by.seed.dt <- subset.dt[,.SD[which.max(validation_auc), .(batch_size,lr)],
                          by=list(experiment,imratio,dataset,seed)]
best.param.dt <- plot.dt[max.by.seed.dt, on=list(batch_size, lr, experiment, imratio, dataset,seed)]
tall.dt <- nc::capture_melt_single(best.param.dt, set = "subtrain|validation|test", "_", metric="auc|our_square_hinge")
tall.dt[,seed.numeric := as.integer(seed)]
wide.dt <- data.table::dcast(tall.dt,epoch+experiment+set+dataset+metric+imratio+seed.numeric~.,list(median,max,min),value.var = "value")

quantile.by.seed <- elapsed.dt[,.(times = quantile(elapsed.time)),by=seed]
quantile.by.seed[[paste("quantiles")]] <- rep(c(0,25,50,75,100),5)
wide.quantile.by.seed <- dcast(quantile.by.seed, seed~quantiles,value.var = "times")