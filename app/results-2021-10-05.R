library(ggplot2)
library(data.table)
results.csv.glob <- "results-2021-10-05/*.csv"
(result.csv.vec <- Sys.glob(results.csv.glob))
wc.cmd <- paste("wc -l", results.csv.glob)
nline.dt <- fread(cmd=wc.cmd, col.names=c("lines", "file"))
some.csv <- nline.dt[lines==101, file]

result.dt <- data.table(result.csv=some.csv)[, {
  fread(result.csv)
}, by=result.csv]

result.dt[, .(
  epochs=.N
), by=.(lr, batch_size, imratio)]

result.dt[, .(
  epochs=.N
), keyby=.(imratio)]

result.dt[, .(
  epochs=.N
), keyby=.(batch_size)]

result.dt[, .(
  epochs=.N
), keyby=.(lr)]

result.dt[, .(
  epochs=.N
), keyby=.(loss_name)]

show.imratio <- 0.01
show.loss <- "square_hinge"
result.tall <- nc::capture_melt_multiple(
  result.dt,
  set="train|test",
  "_",
  column="loss|auc")
some.tall <- result.tall[loss_name==show.loss & imratio==show.imratio]
ggplot()+
  ggtitle(paste0("imratio=", show.imratio, ", loss=", show.loss))+
  facet_grid(batch_size ~ lr, labeller=label_both)+
  geom_line(aes(
    epoch, loss, color=set),
    data=some.tall)

some.min <- some.tall[, .SD[which.min(loss)], by=.(set)]
ggplot()+
  ggtitle(paste0("imratio=", show.imratio, ", loss=", show.loss))+
  facet_grid(. ~ ., labeller=label_both)+
  geom_line(aes(
    epoch, loss, color=set, group=paste(set, lr, batch_size)),
    data=some.tall)+
  geom_point(aes(
    epoch, loss, color=set),
    data=some.min)+
  geom_label(aes(
    epoch, loss, label=sprintf("lr=%e\nbatch_size=%d", lr, batch_size),
    color=set),
    hjust=1,
    vjust=1,
    data=some.min,
    alpha=0.5)

counts.wide <- dcast(result.dt, loss_name + imratio + batch_size ~ lr, length)
counts.tall <- melt(counts.wide, id.vars=c("loss_name", "imratio", "batch_size"), variable.name="lr.chr", value.name="count")
counts.tall[, lr := as.numeric(paste(lr.chr))]
filter.dt <- counts.tall[, {
  missing.dt <- data.table(lr, count)[count==0][which.min(lr)]
  data.table(lr.filter=if(nrow(missing.dt)==0)Inf else missing.dt[["lr"]])
}, by=.(loss_name, imratio, batch_size)]
keep.results <- result.tall[filter.dt, on=.(loss_name, imratio, batch_size), nomatch=0L][lr<lr.filter]
result.min <- keep.results[, 
  .SD[which.min(loss)],
  by=.(set, imratio, loss_name)]

range.dt <- keep.results[, .(
  max.loss = max(loss),
  min.lr=min(lr),
  max.lr=max(lr),
  min.bs=min(batch_size),
  max.bs=max(batch_size)
), by=.(imratio, loss_name)]
range.or.equal <- function(min, max, fmt){
  ifelse(
    min==max,
    sprintf(paste0("=", fmt), min),
    sprintf(paste0("[",fmt,",",fmt,"]"), min, max))
}
ggplot()+
  xlab("epoch")+
  ylab("loss")+
  facet_grid(loss_name ~ imratio, labeller=label_both)+
  geom_text(aes(
    0, max.loss,
    label=sprintf(
      "lr%s batch_size%s",
      range.or.equal(min.lr, max.lr, "%e"),
      range.or.equal(min.bs, max.bs, "%d"))),
    data=range.dt,
    hjust=0,
    vjust=0)+
  geom_line(aes(
    epoch, loss, color=set, group=paste(set, lr, batch_size)),
    data=keep.results)+
  geom_point(aes(
    epoch, loss, color=set),
    data=result.min)+
  geom_label(aes(
    epoch, loss, label=sprintf("lr=%e, batch_size=%d", lr, batch_size),
    color=set),
    hjust=1,
    vjust=1,
    data=result.min,
    alpha=0.5)
