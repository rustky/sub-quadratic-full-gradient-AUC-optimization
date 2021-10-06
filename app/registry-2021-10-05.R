reg.dir <- "registry-2021-10-05"
reg <- batchtools::loadRegistry(reg.dir)

jt <- batchtools::getJobTable(reg=reg)
jt[, .(count=.N), by=.(error, done=!is.na(done))]
## 9 = out of memory (killed).
## 1 = RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.

jt[!is.na(done)]
logs <- batchtools::grepLogs(pattern="Error", reg=reg)
logs[, .(
  jobs=.N
), by=matches]

