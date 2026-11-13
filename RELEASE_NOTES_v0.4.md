# MotorForza Android — Pre-release v0.4

MotorForza é a recompilação nativa de Forza Motorsport (Xbox Original, 2005) para dispositivos Android com backend Vulkan de alta performance e controles táteis integrados.

> [!IMPORTANT]
> **Aviso Legal & Requisitos:** Este projeto NÃO inclui arquivos proprietários, ROMs, ISOs, texturas ou executáveis do jogo original. Você deve fornecer sua própria cópia legítima do jogo (`default.xbe` e arquivos do jogo extraídos em `/storage/emulated/0/Forza` ou selecionar via importador no app).

---

## 🚀 Novidades da Versão 0.4

### 🎮 Gráficos & Renderização 100% Restaurados
- **Correção Crítica de Fog NV2A (D3D8 Vertex Fog Inversion):**
  - Corrigido o cálculo do fator de neblina linear para shaders de vértice (`oFog.x`). Anteriormente, o shader invertia a visibilidade, submergindo pistas, carros e estruturas em escuridão total enquanto apenas o céu renderizava.
  - Carros, pintura metálica, lanternas traseiras, marcas de pneus no asfalto, zebras, arquibancadas, árvores, iluminação de pôr do sol e reflexos agora renderizam com fidelidade visual perfeita.
- **Eliminação de Artefatos Geométricos:**
  - Removido o cache instável de buffers de vértice/índice que causava conflitos em scratches de montagem de geometria.

### ⚡ Otimizações de Desempenho (Fase 1 & Fase 2)
- **Deduplicação de Estado Vulkan Pass-Scoped:**
  - `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdSetViewport`, `vkCmdSetScissor`, `vkCmdBindVertexBuffers` e `vkCmdBindIndexBuffer` agora filtram chamadas redundantes por render pass.
- **Memoização de UBO e Constantes:**
  - Eliminação de uploads redundantes de uniform buffer objects (`const_memo_`) com validação de payload em memória.
- **Otimização de Render Passes e Clears:**
  - Substituição de clears manuais desnecessários por `LOAD_OP_CLEAR` em pass inicial, economizando largura de banda da GPU.
- **Desempenho no dispositivo de teste (Galaxy A54 / GPU Mali-G68):**
  - **30 a 75+ FPS** a 100% de resolução nativa com sombras, iluminação e bloom ativos!

---

## 📦 Arquivo do Pre-release
- `MotorForza-v0.4-prerelease.apk` (~33.7 MB)
