- softmax + topk on a single warp ? not good
- a single SM is doing the GEMV -> we should see if its worth to add more SMs (probbably yes)
- we need to use better typing for kenrels currently they are suboptimal -> hardcoded float4 and threads number 

- the flashmoe.cu -> its literraly the constants , should be written in constants.cu? 
- we need to define better constants namespaces so that we are not having big structures