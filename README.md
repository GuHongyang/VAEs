
Variant models of VAE based on Pytorch
=======

1.VAE

https://arxiv.org/pdf/1312.6114.pdf

<img src="figures/vae.png" width="150">

$$
\log p _ { \boldsymbol { \theta } } \left( \mathbf { x } ^ { ( i ) } \right) \geq \mathcal { L } \left( \boldsymbol { \theta } , \boldsymbol { \phi } ; \mathbf { x } ^ { ( i ) } \right) = \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } \left[ - \log q _ { \phi } ( \mathbf { z } | \mathbf { x } ) + \log p _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { z } ) \right]
$$
$$
\mathcal { L } \left( \boldsymbol { \theta } , \boldsymbol { \phi } ; \mathbf { x } ^ { ( i ) } \right) = - D _ { K L } \left( q _ { \boldsymbol { \phi } } \left( \mathbf { z } | \mathbf { x } ^ { ( i ) } \right) \| p _ { \boldsymbol { \theta } } ( \mathbf { z } ) \right) + \mathbb { E } _ { q _ { \phi } \left( \mathbf { z } | \mathbf { x } ^ { ( i ) } \right) } \left[ \log p _ { \boldsymbol { \theta } } \left( \mathbf { x } ^ { ( i ) } | \mathbf { z } \right) \right]
$$
$$
\widetilde { \mathcal { L } } \left( \boldsymbol { \theta } , \boldsymbol { \phi } ; \mathbf { x } ^ { ( i ) } \right) = - D _ { K L } \left( q _ { \boldsymbol { \phi } } \left( \mathbf { z } | \mathbf { x } ^ { ( i ) } \right) \| p _ { \boldsymbol { \theta } } ( \mathbf { z } ) \right) + \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \left( \log p _ { \boldsymbol { \theta } } \left( \mathbf { x } ^ { ( i ) } | \mathbf { z } ^ { ( i , l ) } \right) \right)
$$
where $\mathbf { z } ^ { ( i , l ) } = g _ { \phi } \left( \boldsymbol { \epsilon } ^ { ( i , l ) } , \mathbf { x } ^ { ( i ) } \right)$ and $\boldsymbol { \epsilon } ^ { ( l ) } \sim p ( \boldsymbol { \epsilon } )$
$$
\mathcal { L } ( \boldsymbol { \theta } , \boldsymbol { \phi } ; \mathbf { X } ) \simeq \widetilde { \mathcal { L } }  \left( \boldsymbol { \theta } , \boldsymbol { \phi } ; \mathbf { X } ^ { M } \right) = \frac { N } { M } \sum _ { i = 1 } ^ { M } \widetilde { \mathcal { L } } \left( \boldsymbol { \theta } , \boldsymbol { \phi } ; \mathbf { x } ^ { ( i ) } \right)
$$
