library(batchtools)
unlink("registry", recursive=TRUE)
reg <- batchtools::makeRegistry("registry")
(slurm.tmpl <- normalizePath(
  "~/slurm-afterok.tmpl",
  mustWork=TRUE))
## 30 hours per 100 epochs
MyFun <- function(batch_size, imratio, lr, num_epochs=100){
  status <- system(paste(
    "python driver.py",
    imratio, lr, num_epochs, batch_size))
  if(status != 0){
    stop("error code ", status)
  }
}
batchtools::batchMap(
   MyFun, args=data.table::CJ(
     batch_size=c(10, 50, 100, 500, 1000, 5000),
     imratio=c(0.001, 0.01, 0.1, 0.5),
     lr=10^seq(-5, 1, by=0.5)
   ), reg=reg)
job.table <- batchtools::getJobTable(reg=reg)
chunks <- data.frame(job.table, chunk=1)
batchtools::submitJobs(chunks, resources=list(
  walltime = 2*24*60*60,#seconds
  memory = 8000,#megabytes per cpu
  ncpus=1,  #>1 for multicore/parallel jobs.
  ntasks=1, #>1 for MPI jobs.
  chunks.as.arrayjobs=TRUE), reg=reg)

jt <- batchtools::getJobTable(reg=reg)
jt[!is.na(error)]
