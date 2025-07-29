# Auto_trader
# AutoTrader IA LSTM Ultra-Precisão

Este projeto é um sistema avançado de auto-trading para ouro, utilizando IA (LSTM), Streamlit e técnicas institucionais de análise de mercado.

## Como rodar

1. **Crie um ambiente virtual (opcional, recomendado):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Instale as dependências:**

   ```bash
   pip install streamlit pandas numpy plotly tensorflow scikit-learn ta
   ```

3. **Execute o sistema:**

   ```bash
   streamlit run app.py
   ```

4. **Acesse o dashboard:**
   Abra o navegador em `http://localhost:8501`

## Requisitos do arquivo CSV

- Colunas obrigatórias: `open`, `high`, `low`, `close`, `volume`
- Timestamp: opcional, mas recomendado
- Mínimo: 70+ registros
- Dados limpos, sem valores nulos

## Principais funcionalidades

- Interface interativa com Streamlit
- Predição multi-sequência com IA LSTM
- Indicadores técnicos avançados
- Sistema de alertas multi-canal
- Backtesting institucional
- Gestão de risco automatizada

## Observações

- O modo "full_auto" pode executar ordens reais. Use com cautela!
- O sistema pode exigir arquivos de modelo (`model_lstm_optimized.keras`) e scaler (`scaler_robust.npy`).

## Licença

Este projeto é de autoria de ralfsantosdev para fins educacionais e experimentais.
