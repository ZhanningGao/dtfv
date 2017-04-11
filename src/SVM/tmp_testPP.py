import pp

job_server = pp.Server(ncpus=10)

ncpus = job_server.get_ncpus()

print("number of cpu: %d"%(ncpus))

job_server.destroy()

