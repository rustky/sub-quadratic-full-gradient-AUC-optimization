ex_1 <- "results-2022-01-31/"
ex_2 <- "results-2022-01-26/"
ex_3 <- "results-2022-01-25/"
ex_4 <- "results-2022-01-24/"

library(ggplot2)
library(plyr)
library(data.table)
setwd('~/Documents/monsoon')
experiment.desc <- list("experiment1"= "Squared Hinge + LBFGS",
                        "experiment2"= "Squared Hinge + SGD",
                        "experiment3"= "LIBAUC + SGD",
                        "experiment4"= "LIBAUC + PESG")
experiment.list  <- c("experiment1","experiment2",
                       "experiment3", "experiment4")
result.dt.list <- list()
plot.frame.list <- list()
for(experiment in experiment.list){
  results.csv.glob <- paste(experiment,"*.csv",sep='/')
  (result.csv.vec <- Sys.glob(results.csv.glob))
  wc.cmd <- paste("wc -l", results.csv.glob)
  nline.dt <- fread(cmd=wc.cmd, col.names=c("lines", "file"))
  some.csv <- nline.dt[lines>6][file!="total", file]
  split_csv <- strsplit(some.csv, '-|=')
  split.table <- data.table(do.call(rbind, split_csv))
  set.col <- split.table[,'V12']
  set.vec <- rep(unlist(set.col[,"V12"]),each = 100)
  result.dt <- data.table(result.csv=some.csv)[, {
    fread(result.csv)
  }, by=result.csv]
  result.dt$dataset = set.vec
  result.dt$experiment = experiment.desc[[paste(experiment)]]
  result.dt.list[[paste(experiment)]] <- result.dt
  set.to.show = 'CAT_VS_DOG'
  ratio.to.show = 0.5
  bool.vec <- which((result.dt[,'dataset'] == set.to.show) &
                      (result.dt[,'imratio'] == ratio.to.show))
  plot.frame.list[[paste(experiment)]] <- result.dt[bool.vec,]
}

plot.dt <- data.table(do.call(rbind.fill, plot.frame.list))

ggplot() +
  geom_line(data=plot.dt,
            aes(epoch,test_auc, color= experiment))+
  facet_grid(lr ~ batch_size)+
  ggtitle(paste(set.to.show,ratio.to.show, sep=" "))

ggplot() +
  geom_line(data=plot.dt,
            aes(epoch,test_our_square_hinge, color=experiment))+
  facet_grid(lr ~ batch_size)+
  ggtitle(paste(set.to.show,ratio.to.show, sep=" "))

ggplot()+
  geom_line(data=plot.dt,
            aes(epoch,test_libauc_loss, color=experiment))+
  facet_grid(lr ~ batch_size)+
  ggtitle(paste(set.to.show,ratio.to.show, sep=" "))

