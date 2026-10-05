# olmoforge.routing.rowwise

`RowwiseExpertParallel` 把路由到的 token 直写进 per-expert 缓冲，省掉显式的
gather-permute-scatter 重排。
