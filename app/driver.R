library(batchtools)
out_dir <- strftime(Sys.time(), "results-%Y-%m-%d")

dir.create(out_dir)
reg.dir <- sub("results", "registry", out_dir)
unlink(reg.dir, recursive=TRUE)
reg <- batchtools::makeRegistry(reg.dir)
## 30 hours per 100 epochs
MyFun <- function(batch_size, imratio, loss_name, lr, out_dir){
  status <- system(paste(
    "python driver.py",
    batch_size, imratio, loss_name, lr, out_dir))
  if(status != 0){
    stop("error code ", status)
  }
}
batchtools::batchMap(
  MyFun,
  more.args=list(
    out_dir=out_dir
  ), 
  args=data.table::CJ(
    loss_name=c("square_hinge"),
    batch_size=c(10, 50, 100, 500, 1000, 5000),
    imratio=c(0.001, 0.01, 0.1, 0.5),
    lr=10^seq(-4, 0, by=0.5)
  ), reg=reg)
job.table <- batchtools::getJobTable(reg=reg)
chunks <- data.frame(job.table, chunk=1)
batchtools::submitJobs(chunks, resources=list(
  walltime = 2*24*60*60,#seconds
  memory = 16000,#megabytes per cpu
  ncpus=1,  #>1 for multicore/parallel jobs.
  ntasks=1, #>1 for MPI jobs.
  chunks.as.arrayjobs=TRUE), reg=reg)

reg.dir.vec <- Sys.glob("registry*")
reg.dir <- reg.dir.vec[length(reg.dir.vec)]
reg <- batchtools::loadRegistry(reg.dir)

jt <- batchtools::getJobTable(reg=reg)
jt[, .(count=.N), by=error]
## 9 = out of memory (killed).
## 1 = RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.

jt[!is.na(done)]
logs <- batchtools::grepLogs(pattern="Error", reg=reg)
logs[, .(
  jobs=.N
), by=matches]

