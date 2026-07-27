# FlashMoE

Operador Mixture of Experts de kernel único para GPUs NVIDIA, inspirado en el [paper FlashDMoE](https://arxiv.org/abs/2506.04667) y en [Flash-Moe, Piotr.k](https://github.com/makora-ai/flash-moe).

En este repositorio he recreado el paper FlashDMoE, donde el enfoque es una planificación basada en actores que define roles concretos para los SMs. Este enfoque resultó generar muchos problemas de planificación para el kernel en sí (la mayoría intenté solucionarlos mediante algunos pequeños trucos). El problema principal con la planificación actual es que un solo hilo de un warp se dedica a enviar tareas a los bloques Worker.
Resulta que hay muchas esperas (según mi herramienta de trazado utilizada).
Este enfoque es experimental y se beneficiaría de una configuración multi-GPU donde la latencia de la comunicación de la GPU sea el cuello de botella.
Un enfoque mejor (sin experimentar aún) sería utilizar kernels separados.

Trucos utilizados y aprendizajes

A nivel de Kernel:
- Topk primero y luego softmax -> no hacer softmax y luego topk cuando solo necesitamos softmax para estabilidad numérica :D
- Gate y up fusionados para la FFN
- Prefetch + ILP para el GEMV. La memoria compartida fue inútil
- Pensar en la operación y probar micro-benchmarks para tener una mejor sensación de los problemas

Visión general:
- Pensar en formas de analizar el problema desde una perspectiva diferente donde el kernel sea una parte pequeña y el sistema más grande sea el de tareas (abriendo puertas y horizontes al DLC basado en tareas)
- Analizar el cómputo del problema para optimizarlo.
- Comprender por qué es importante un mejor mapeo del hardware (mi enfoque actual no era eficiente desde esta perspectiva)

## Trazados de Planificación (Scheduling Traces)

### Enfoque actual
Utiliza un round-robin `next_w` para encontrar el siguiente SM worker listo y verificar su estado antes de asignar trabajo. Esto evita el cuello de botella de un solo hilo despachando todas las tareas secuencialmente y distribuye la carga de planificación de manera más uniforme entre los workers. También distribuimos el enrutamiento inicial entre los SMs. Sin embargo, seguimos viendo burbujas grandes en la fase GEMV_DOWN.

![Current scheduling](images/final_stage.png)

### Otros enfoques

1. Cola simple. Un solo hilo maneja todo el despacho de tareas secuencialmente. La mayoría de los SMs worker terminan esperando inactivos.

![Initial scheduling](images/initial.png)

2. Múltiples warps de planificación. Utiliza más warps relacionados con la planificación para reducir el cuello de botella del despacho.

![Different scheduling showcase](images/differente_scheduling_%29showcase.png)

Dirigido a **decodificación de token único** a través de una capa MoE (configuración Qwen3-30B-A3B) en una RTX 4070.

## Arquitectura

Un kernel persistente con un SO embebido:

- Bloque OS (1 SM) planificador (asigna tareas a los workers vía doorbells)
- Bloques Worker (45 SMs) consultan los doorbells, ejecutan tiles de FFN, envían tareas de seguimiento al completarse el fan-in

```
Bootstrap → push FFN1 tiles → Scheduler → doorbells → Workers
                                    ↑                      │
                                    └── FFN2 tiles ←───────┘
```

## Configuración

Hardcodeado en `csrc/flashmoe.cuh` para coincidir con Qwen3-30B-A3B:

| Parámetro | Valor |
|-----------|-------|
| HIDDEN_SIZE | 2048 |
| MOE_INTERMEDIATE_SIZE | 768 |
| NUM_EXPERTS | 128 |
| TOP_K | 8 |
| Activación | SiLU (SwiGLU) |

## Benchmark

GPU Laptop RTX 4070, config Qwen3-30B-A3B, T=1, fp16, 200 iteraciones, 50 de calentamiento (warmup).

| Condición | Reloj GPU | vLLM (ms) | FlashMoE (ms) | Speedup |
|-----------|-----------|-----------|---------------|---------|
| Reloj bloqueado | 2700 MHz | 0.348 | 0.413 | 0.84x |
| Batería (desbloqueado) | ~3100 MHz boost | 0.628 | 0.493 | 1.27x |
| Cargador (desbloqueado) | throttling | 0.342 | 0.413 | 0.83x |

El resultado honesto con relojes bloqueados es **0.84x**.

### Lanzamientos de kernel: 1 vs 9

vLLM necesita 9 lanzamientos de kernel separados por cada pase forward de MoE (topkGating, moe_align_block_size, count_and_sort, gemvx, 2x fused_moe_kernel, act_and_mul, reduce, memcpy). FlashMoE lo hace en 1.

![Kernel launches comparison](images/kernel_launches.png)

### Aprendizajes del benchmarking

- Batería vs cargador da resultados diferentes. vLLM pasó de 0.628ms a 0.342ms mientras que el nuestro apenas cambió (0.493 a 0.413ms).
- Bloquear el reloj de la GPU (`nvidia-smi -lgc 2700,2700`) para obtener resultados reproducibles.
- Más calentamiento (50 en lugar de 10) para alcanzar el estado térmico estable antes de medir.
- Estado separado por benchmark. Se corrigió un error de contador de entrada compartido y se añadió `torch.cuda.empty_cache()` entre ejecuciones.
