library(data.table)
library(ggplot2)
library(plyr)
setwd('~/Documents/monsoon/')
filenames <- list("ours"= 'pretrained=True-loss_name=libauc_loss-lr=0.1-batch_size=128-imratio=0.1-dataset=CIFAR10-model=ResNet20.csv',
               "libauc"= 'LIBAUC-Tutorial')
title <- "lr: 0.1 imratio: 0.1 batchsize: 128 dataset: CIFAR10 model: ResNet20 Opt: PESG"
dt_ours <- data.table(read.csv(filenames[['ours']], header=TRUE, sep = '\t'))
dt_libauc <- data.table(read.csv(filenames[['libauc']], sep=' ', header= FALSE))
colnames(dt_libauc) <- c('epoch', 'train_libauc_loss','train_auc','test_libauc_loss','test_auc','lr')
dt_ours[, 'code_base' := 'ours' ]
dt_libauc[, 'code_base' := 'libauc' ]
plot_df = rbind.fill(dt_ours,dt_libauc)
ggplot(plot_df)+
  geom_line(aes(epoch, test_auc,color=code_base))+
  ggtitle(title)

ggplot(plot_df)+
  geom_line(aes(epoch, train_auc,color=code_base))+
  ggtitle(title)

ggplot(plot_df)+
  geom_line(aes(epoch, train_libauc_loss, color=code_base))+
  ggtitle(title)

ggplot(plot_df)+
  geom_line(aes(epoch, test_libauc_loss, color=code_base))+
  ggtitle(title)

folder = 'LBFGS-test'
results.csv.glob <- paste(folder,"*.csv",sep='/')
(result.csv.vec <- Sys.glob(results.csv.glob))
wc.cmd <- paste("wc -l", results.csv.glob)
nline.dt <- fread(cmd=wc.cmd, col.names=c("lines", "file"))
some.csv <- nline.dt[lines>1][file!="total", file]
result.dt <- data.table(result.csv=some.csv)[, {
  fread(result.csv)
}, by=result.csv]

tall.dt <- melt(
  result.dt,
  measure.vars = c('train_our_square_hinge','test_our_square_hinge'),
  value.name = 'loss',
  variable.name = 'set'
)
ggplot(tall.dt)+
  geom_line(aes(epoch, loss, color=set))+
  facet_grid(.~lr)

ggplot(result.dt)+
  geom_line(aes(epoch, test_auc))+
  facet_grid(.~lr)

test.csv <-"pretrained=False-loss_name=weighted_logistic_loss-lr=0.01-batch_size=100-imratio=0.1-dataset=STL10-model=ResNet20-optimizer=SGD-seed=1.csv"
result.dt <- data.table(result.csv=test.csv)[, {
  fread(result.csv)
}, by=result.csv]

tall.dt <- melt(
  result.dt,
  measure.vars = c('subtrain_weighted_log_loss','validation_weighted_log_loss',
                   'test_weighted_log_loss'),
  value.name = 'loss',
  variable.name = 'set'
)

gg <- ggplot(tall.dt)+
  geom_line(aes(epoch,loss,color = set))
gg
