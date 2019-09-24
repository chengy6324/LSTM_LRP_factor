import numpy as np
import pandas as pd
from sklearn import linear_model
from utils import sub_m

# price_data = pd.read_excel('./data/TRD_Mnth.xlsx')  # 个股月收盘价、月交易股数
# # print(price_data)
# price_data = price_data.iloc[2:, :]
# price_data.reset_index(drop=True, inplace=True)
#
# monthly_price = pd.DataFrame()  # 个股月收盘价
# # sz50_list = []
# for name, group in price_data.groupby('Stkcd'):
#     temp_close = group.drop(['Stkcd', 'Mnshrtrd'], axis=1)  # 删除剩余两列
#     temp_close.set_index('Trdmnt', inplace=True)
#     temp_close.columns = [name]
#     monthly_price = pd.concat([monthly_price, temp_close], axis=1, sort=True)
# # print(monthly_price)
#
# start_date = '2001-08'
# monthly_price = monthly_price.loc[start_date:, :]
# monthly_price.dropna(axis=1, how='any', inplace=True)
# print(monthly_price)
# pd.DataFrame(monthly_price.columns).to_excel('./data/20支股票代码.xlsx', header=False, index=False)

# 20支股票代码
stock_code = pd.read_excel('./data/20支股票代码.xlsx', header=None)
stock_code_list = list(stock_code.iloc[:, 0].values)

start_date = '2001-08'

# 沪深300成份股月度收盘价数据
price_m = pd.read_excel('./data/沪深300成份股收盘价.xlsx')
price_m.columns = list(price_m.iloc[2, :].values)
price_m = price_m.iloc[3:, :]
price_m.loc[:, 'Date'] = price_m.loc[:, 'Date'].apply(sub_m)
price_m.set_index(['Date'], drop=True, inplace=True)
price_m = price_m.loc[start_date:, stock_code_list]
# print(price_m)

return_m = np.log(np.array(price_m.iloc[1:, :].values / price_m.iloc[:-1, :].values, dtype=float))
return_m = pd.DataFrame(return_m)
return_m.columns = price_m.columns
return_m.index = list(price_m.index)[1:]
# print(return_m)

# 风险因子
# 60VOL
VOL60 = pd.DataFrame()
month_lag = 60
for i in range(month_lag, return_m.shape[0]):
    # print(return_m.iloc[i - month_lag:i, :])
    temp_std = return_m.iloc[i - month_lag:i, :].std()
    VOL60 = pd.concat([VOL60, pd.DataFrame([temp_std.values])], axis=0)
VOL60.columns = return_m.columns
VOL60.index = list(return_m.index)[month_lag:]
# print(VOL60)

# BETA
# 上证综指月度收盘价数据
price_m_1 = pd.read_excel('./data/上证综指收盘价.xlsx')
price_m_1.columns = list(price_m_1.iloc[2, :].values)
price_m_1 = price_m_1.iloc[3:, :]
price_m_1.loc[:, 'Date'] = price_m_1.loc[:, 'Date'].apply(sub_m)
price_m_1.set_index(['Date'], drop=True, inplace=True)
price_m_1 = price_m_1.loc[list(price_m.index)[0]:, :]
# print(price_m_1)

return_m_1 = np.log(np.array(price_m_1.iloc[1:, :].values / price_m_1.iloc[:-1, :].values, dtype=float))
return_m_1 = pd.DataFrame(return_m_1)
return_m_1.columns = price_m_1.columns
return_m_1.index = list(price_m_1.index)[1:]
# print(return_m_1)

beta_month_lag = 60
BETA = pd.DataFrame()
for i in range(beta_month_lag, return_m.shape[0]):
    temp_risk_premium = return_m_1.iloc[i - beta_month_lag:i, :]
    beta_list = []
    for j in range(return_m.shape[1]):
        temp_return = return_m.iloc[i - beta_month_lag:i, j]
        model = linear_model.LinearRegression()
        model.fit(np.array(temp_risk_premium.values, dtype=float).reshape(-1, 1), np.array(temp_return.values, dtype=float).reshape(-1, 1))
        beta_list.append(model.coef_[0][0])
    BETA = pd.concat([BETA, pd.DataFrame([beta_list])], axis=0)
BETA.columns = return_m.columns
BETA.index = list(return_m.index)[beta_month_lag:]
# print(BETA)

# SKEW
SKEW = pd.DataFrame()
skew_month_lag = 60
for i in range(skew_month_lag, return_m.shape[0]):
    # print(return_m.iloc[i-month_lag:i, :])
    temp_skew = return_m.iloc[i - skew_month_lag:i, :].skew()
    SKEW = pd.concat([SKEW, pd.DataFrame([temp_skew])], axis=0)
SKEW.columns = return_m.columns
SKEW.index = list(return_m.index)[skew_month_lag:]
# print(SKEW)

# 质量因子（只有季度数据）

# 动量因子
# 12-1MOM
MOM12_1_re = pd.DataFrame()
long_mom = 12
short_mom = 1
for i in range(long_mom, return_m.shape[0]):
    # print(return_m.iloc[i - long_mom:i, :])
    # print(return_m.iloc[i - 1, :])
    temp_long_mom = np.array(return_m.iloc[i - long_mom:i, :].sum())
    temp_short_mom = np.array(return_m.iloc[i - short_mom:i, :].sum())
    temp_sub = temp_long_mom - temp_short_mom
    MOM12_1_re = pd.concat([MOM12_1_re, pd.DataFrame([temp_sub])], axis=0)
MOM12_1_re.columns = return_m.columns
MOM12_1_re.index = list(return_m.index)[long_mom:]
# print(MOM12_1_re)

# 1MOM
MOM_1_re = pd.DataFrame()
short_mom = 1
for i in range(short_mom, return_m.shape[0]):
    temp_short_mom = np.array(return_m.iloc[i - short_mom:i, :].sum())
    MOM_1_re = pd.concat([MOM_1_re, pd.DataFrame([temp_short_mom])], axis=0)
MOM_1_re.columns = return_m.columns
MOM_1_re.index = list(return_m.index)[short_mom:]
# print(MOM_1_re)

# 60MOM
MOM_60_re = pd.DataFrame()
long_mom = 60
for i in range(long_mom, return_m.shape[0]):
    temp_long_mom = np.array(return_m.iloc[i - long_mom:i, :].sum())
    MOM_60_re = pd.concat([MOM_60_re, pd.DataFrame([temp_long_mom])], axis=0)
MOM_60_re.columns = return_m.columns
MOM_60_re.index = list(return_m.index)[long_mom:]
# print(MOM_60_re)

# 价值因子
# PSR
PSR = pd.read_excel('./data/20支股票市销率.xlsx')
PSR.columns = list(PSR.iloc[2, :].values)
PSR = PSR.iloc[3:, :]
PSR.loc[:, 'Date'] = PSR.loc[:, 'Date'].apply(sub_m)
PSR.set_index(['Date'], drop=True, inplace=True)
# print(PSR)

# PER
# PBR
PBR = pd.read_excel('./data/20支股票市净率.xlsx')
PBR.columns = list(PBR.iloc[2, :].values)
PBR = PBR.iloc[3:, :]
PBR.loc[:, 'Date'] = PBR.loc[:, 'Date'].apply(sub_m)
PBR.set_index(['Date'], drop=True, inplace=True)
PBR.dropna(how='any', axis=1, inplace=True)
# print(PBR)

# PCFR

# 规模因子
# CAP
CAP = pd.read_excel('./data/20支股票总市值.xlsx')
CAP.columns = list(CAP.iloc[2, :].values)
CAP = CAP.iloc[3:, :]
CAP.loc[:, 'Date'] = CAP.loc[:, 'Date'].apply(sub_m)
CAP.set_index(['Date'], drop=True, inplace=True)
CAP.dropna(how='any', axis=1, inplace=True)
CAP_array = np.array(CAP.values, dtype=float)
CAP_array_log = np.log(CAP_array)
CAP_log = pd.DataFrame(CAP_array_log)
CAP_log.columns = CAP.columns
CAP_log.index = list(CAP.index)
# print(CAP_log)

# ILLIQ
trading_volume = pd.read_excel('./data/20支股票成交量.xlsx', index_col=0)
trading_volume.reset_index(drop=False, inplace=True)
trading_volume.loc[:, 'index'] = trading_volume.loc[:, 'index'].apply(sub_m)
trading_volume.set_index(['index'], drop=True, inplace=True)
# print(trading_volume)

ILLIQ_re = pd.DataFrame()
ILLIQ_month_lag = 60
for i in range(ILLIQ_month_lag, return_m.shape[0]):
    temp_returns = return_m.iloc[i - ILLIQ_month_lag:i, :]
    temp_trading_volume = trading_volume.iloc[i - ILLIQ_month_lag:i, :]
    temp_di = abs(temp_returns) / temp_trading_volume
    temp_di_mean = temp_di.mean()
    ILLIQ_re = pd.concat([ILLIQ_re, pd.DataFrame([temp_di_mean.values])], axis=0)
ILLIQ_re.columns = return_m.columns
ILLIQ_re.index = list(return_m.index)[ILLIQ_month_lag:]
# print(ILLIQ_re)

# 统一日期
basic_date = list(VOL60.index)

# 风险因子
# 60VOL
VOL60_mon = VOL60.loc[basic_date, :]
# BETA
BETA_mon = BETA.loc[basic_date, :]
# SKEW
SKEW_mon = SKEW.loc[basic_date, :]

# 质量因子
# 12-1MOM
MOM12_1_mon = MOM12_1_re.loc[basic_date, :]
# 1MOM
MOM_1_mon = MOM_1_re.loc[basic_date, :]
# 60MOM
MOM_60_mon = MOM_60_re.loc[basic_date, :]

# 价值因子
# PSR
PSR_mon = PSR.loc[basic_date, :]
# PBR
PBR_mon = PBR.loc[basic_date, :]

# 规模因子
# CAP
CAP_mon = CAP_log.loc[basic_date, :]
# ILLIQ
ILLIQ_mon = ILLIQ_re.loc[basic_date, :]

# 因子顺序
# 60VOL BETA SKEW 12-1MOM 1MOM 60MOM PSR PBR CAP ILLIQ

# 将因子数据写入excel文件，每类因子占一个Sheet
write = pd.ExcelWriter('./data/monthly_factor.xlsx')
VOL60_mon.to_excel(write, sheet_name='60VOL')
BETA_mon.to_excel(write, sheet_name='BETA')
SKEW_mon.to_excel(write, sheet_name='SKEW')

MOM12_1_mon.to_excel(write, sheet_name='12-1MOM')
MOM_1_mon.to_excel(write, sheet_name='1MOM')
MOM_60_mon.to_excel(write, sheet_name='60MOM')

PSR_mon.to_excel(write, sheet_name='PSR')
PBR_mon.to_excel(write, sheet_name='PBR')

CAP_mon.to_excel(write, sheet_name='CAP')
ILLIQ_mon.to_excel(write, sheet_name='ILLIQ')
write.save()

# 对应日期的月度收益率
monthly_return_mon = return_m.loc[basic_date, :]
monthly_return_mon.to_excel('./data/monthly_return.xlsx')
