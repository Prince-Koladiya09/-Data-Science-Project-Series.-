{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "d4e8e0c9-6188-46ae-80d9-996eb214dde0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV\n",
    "from sklearn.linear_model import LinearRegression,LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.preprocessing import StandardScaler\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9f856016-14c3-4d6a-a603-990c517f5b36",
   "metadata": {},
   "source": [
    "# Load the dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c76dd31f-97cb-4943-a652-b02d40e719c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('infolimpioavanzadoTarget.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "8bd59b51-49e3-4d4e-9b10-0af79ebcec62",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7781, 1285)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d61c24bd-3d36-4e62-abc8-a365919bdee5",
   "metadata": {},
   "source": [
    "# Display basic statistics of the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "d99f3644-4b36-48a6-9827-4c2c1c5a6b8d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>open</th>\n",
       "      <th>high</th>\n",
       "      <th>low</th>\n",
       "      <th>close</th>\n",
       "      <th>adjclose</th>\n",
       "      <th>volume</th>\n",
       "      <th>RSIadjclose15</th>\n",
       "      <th>RSIvolume15</th>\n",
       "      <th>RSIadjclose25</th>\n",
       "      <th>RSIvolume25</th>\n",
       "      <th>...</th>\n",
       "      <th>high-15</th>\n",
       "      <th>K-15</th>\n",
       "      <th>D-15</th>\n",
       "      <th>stochastic-k-15</th>\n",
       "      <th>stochastic-d-15</th>\n",
       "      <th>stochastic-kd-15</th>\n",
       "      <th>volumenrelativo</th>\n",
       "      <th>diff</th>\n",
       "      <th>INCREMENTO</th>\n",
       "      <th>TARGET</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>7781.000000</td>\n",
       "      <td>7781.000000</td>\n",
       "      <td>7781.000000</td>\n",
       "      <td>7781.000000</td>\n",
       "      <td>7781.000000</td>\n",
       "      <td>7.781000e+03</td>\n",
       "      <td>7316.000000</td>\n",
       "      <td>7316.000000</td>\n",
       "      <td>7006.000000</td>\n",
       "      <td>7006.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>7347.000000</td>\n",
       "      <td>7262.000000</td>\n",
       "      <td>7194.000000</td>\n",
       "      <td>7262.000000</td>\n",
       "      <td>7194.000000</td>\n",
       "      <td>7194.000000</td>\n",
       "      <td>7566.000000</td>\n",
       "      <td>7626.000000</td>\n",
       "      <td>7626.000000</td>\n",
       "      <td>7781.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>34.990220</td>\n",
       "      <td>35.655999</td>\n",
       "      <td>34.301243</td>\n",
       "      <td>34.964414</td>\n",
       "      <td>34.483147</td>\n",
       "      <td>7.586022e+05</td>\n",
       "      <td>46.817434</td>\n",
       "      <td>49.814790</td>\n",
       "      <td>46.966016</td>\n",
       "      <td>49.898659</td>\n",
       "      <td>...</td>\n",
       "      <td>37.947291</td>\n",
       "      <td>18.673824</td>\n",
       "      <td>18.704812</td>\n",
       "      <td>18.673824</td>\n",
       "      <td>18.704812</td>\n",
       "      <td>0.298413</td>\n",
       "      <td>inf</td>\n",
       "      <td>-0.259186</td>\n",
       "      <td>-2.674224</td>\n",
       "      <td>0.183010</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>99.841502</td>\n",
       "      <td>101.451058</td>\n",
       "      <td>98.073945</td>\n",
       "      <td>99.790823</td>\n",
       "      <td>98.603879</td>\n",
       "      <td>3.934491e+06</td>\n",
       "      <td>11.672838</td>\n",
       "      <td>5.002664</td>\n",
       "      <td>8.760961</td>\n",
       "      <td>3.420371</td>\n",
       "      <td>...</td>\n",
       "      <td>107.340294</td>\n",
       "      <td>75.723295</td>\n",
       "      <td>74.210933</td>\n",
       "      <td>75.723295</td>\n",
       "      <td>74.210933</td>\n",
       "      <td>14.661948</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.334250</td>\n",
       "      <td>268.268134</td>\n",
       "      <td>0.386699</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.410000</td>\n",
       "      <td>0.435000</td>\n",
       "      <td>0.405000</td>\n",
       "      <td>0.408000</td>\n",
       "      <td>0.408000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>6.837461</td>\n",
       "      <td>35.303213</td>\n",
       "      <td>17.693637</td>\n",
       "      <td>39.520876</td>\n",
       "      <td>...</td>\n",
       "      <td>0.510000</td>\n",
       "      <td>-668.212635</td>\n",
       "      <td>-626.263336</td>\n",
       "      <td>-668.212635</td>\n",
       "      <td>-626.263336</td>\n",
       "      <td>-211.219037</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-90.538818</td>\n",
       "      <td>-23399.465955</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>4.050000</td>\n",
       "      <td>4.130000</td>\n",
       "      <td>3.980000</td>\n",
       "      <td>4.030000</td>\n",
       "      <td>3.960000</td>\n",
       "      <td>1.080000e+04</td>\n",
       "      <td>38.946316</td>\n",
       "      <td>47.182234</td>\n",
       "      <td>40.954487</td>\n",
       "      <td>48.266978</td>\n",
       "      <td>...</td>\n",
       "      <td>4.565000</td>\n",
       "      <td>6.153839</td>\n",
       "      <td>8.336837</td>\n",
       "      <td>6.153839</td>\n",
       "      <td>8.336837</td>\n",
       "      <td>-6.585432</td>\n",
       "      <td>0.637237</td>\n",
       "      <td>-0.417873</td>\n",
       "      <td>-4.494383</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>10.080000</td>\n",
       "      <td>10.110000</td>\n",
       "      <td>10.005000</td>\n",
       "      <td>10.080000</td>\n",
       "      <td>10.061000</td>\n",
       "      <td>8.406000e+04</td>\n",
       "      <td>46.259711</td>\n",
       "      <td>48.356834</td>\n",
       "      <td>46.459477</td>\n",
       "      <td>48.961162</td>\n",
       "      <td>...</td>\n",
       "      <td>10.640000</td>\n",
       "      <td>28.484828</td>\n",
       "      <td>28.478797</td>\n",
       "      <td>28.484828</td>\n",
       "      <td>28.478797</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.025000</td>\n",
       "      <td>-0.304004</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>24.350000</td>\n",
       "      <td>24.500000</td>\n",
       "      <td>24.080000</td>\n",
       "      <td>24.250000</td>\n",
       "      <td>22.466007</td>\n",
       "      <td>6.724000e+05</td>\n",
       "      <td>54.061089</td>\n",
       "      <td>50.902284</td>\n",
       "      <td>52.289893</td>\n",
       "      <td>50.527067</td>\n",
       "      <td>...</td>\n",
       "      <td>25.170000</td>\n",
       "      <td>59.688404</td>\n",
       "      <td>58.664021</td>\n",
       "      <td>59.688404</td>\n",
       "      <td>58.664021</td>\n",
       "      <td>6.726947</td>\n",
       "      <td>1.655385</td>\n",
       "      <td>0.240000</td>\n",
       "      <td>2.812552</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>795.739990</td>\n",
       "      <td>799.359985</td>\n",
       "      <td>784.960022</td>\n",
       "      <td>797.489990</td>\n",
       "      <td>783.376221</td>\n",
       "      <td>1.615550e+08</td>\n",
       "      <td>96.365095</td>\n",
       "      <td>99.622735</td>\n",
       "      <td>91.023108</td>\n",
       "      <td>97.782293</td>\n",
       "      <td>...</td>\n",
       "      <td>799.359985</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>198.156313</td>\n",
       "      <td>inf</td>\n",
       "      <td>120.256775</td>\n",
       "      <td>425.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 1283 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "              open         high          low        close     adjclose  \\\n",
       "count  7781.000000  7781.000000  7781.000000  7781.000000  7781.000000   \n",
       "mean     34.990220    35.655999    34.301243    34.964414    34.483147   \n",
       "std      99.841502   101.451058    98.073945    99.790823    98.603879   \n",
       "min       0.410000     0.435000     0.405000     0.408000     0.408000   \n",
       "25%       4.050000     4.130000     3.980000     4.030000     3.960000   \n",
       "50%      10.080000    10.110000    10.005000    10.080000    10.061000   \n",
       "75%      24.350000    24.500000    24.080000    24.250000    22.466007   \n",
       "max     795.739990   799.359985   784.960022   797.489990   783.376221   \n",
       "\n",
       "             volume  RSIadjclose15  RSIvolume15  RSIadjclose25  RSIvolume25  \\\n",
       "count  7.781000e+03    7316.000000  7316.000000    7006.000000  7006.000000   \n",
       "mean   7.586022e+05      46.817434    49.814790      46.966016    49.898659   \n",
       "std    3.934491e+06      11.672838     5.002664       8.760961     3.420371   \n",
       "min    0.000000e+00       6.837461    35.303213      17.693637    39.520876   \n",
       "25%    1.080000e+04      38.946316    47.182234      40.954487    48.266978   \n",
       "50%    8.406000e+04      46.259711    48.356834      46.459477    48.961162   \n",
       "75%    6.724000e+05      54.061089    50.902284      52.289893    50.527067   \n",
       "max    1.615550e+08      96.365095    99.622735      91.023108    97.782293   \n",
       "\n",
       "       ...      high-15         K-15         D-15  stochastic-k-15  \\\n",
       "count  ...  7347.000000  7262.000000  7194.000000      7262.000000   \n",
       "mean   ...    37.947291    18.673824    18.704812        18.673824   \n",
       "std    ...   107.340294    75.723295    74.210933        75.723295   \n",
       "min    ...     0.510000  -668.212635  -626.263336      -668.212635   \n",
       "25%    ...     4.565000     6.153839     8.336837         6.153839   \n",
       "50%    ...    10.640000    28.484828    28.478797        28.484828   \n",
       "75%    ...    25.170000    59.688404    58.664021        59.688404   \n",
       "max    ...   799.359985   100.000000   100.000000       100.000000   \n",
       "\n",
       "       stochastic-d-15  stochastic-kd-15  volumenrelativo         diff  \\\n",
       "count      7194.000000       7194.000000      7566.000000  7626.000000   \n",
       "mean         18.704812          0.298413              inf    -0.259186   \n",
       "std          74.210933         14.661948              NaN     7.334250   \n",
       "min        -626.263336       -211.219037         0.000000   -90.538818   \n",
       "25%           8.336837         -6.585432         0.637237    -0.417873   \n",
       "50%          28.478797          0.000000         1.000000    -0.025000   \n",
       "75%          58.664021          6.726947         1.655385     0.240000   \n",
       "max         100.000000        198.156313              inf   120.256775   \n",
       "\n",
       "         INCREMENTO       TARGET  \n",
       "count   7626.000000  7781.000000  \n",
       "mean      -2.674224     0.183010  \n",
       "std      268.268134     0.386699  \n",
       "min   -23399.465955     0.000000  \n",
       "25%       -4.494383     0.000000  \n",
       "50%       -0.304004     0.000000  \n",
       "75%        2.812552     0.000000  \n",
       "max      425.000000     1.000000  \n",
       "\n",
       "[8 rows x 1283 columns]"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8b17f34-8634-4dd4-98c9-9ee33d963e1b",
   "metadata": {},
   "source": [
    "# Checking the data types and missing values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ccdeb76-b307-4d3a-ab34-4c5e0e584fb8",
   "metadata": {},
   "source": [
    "- > There are many NULL values in the dataset as following."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "4518eee3-e957-4ddd-93de-b7774d15ca46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "date                  0\n",
       "open                  0\n",
       "high                  0\n",
       "low                   0\n",
       "close                 0\n",
       "                   ... \n",
       "stochastic-kd-15    587\n",
       "volumenrelativo     215\n",
       "diff                155\n",
       "INCREMENTO          155\n",
       "TARGET                0\n",
       "Length: 1285, dtype: int64"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "aca42b12-210d-4377-b0ec-e47aab3af32e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>open</th>\n",
       "      <th>high</th>\n",
       "      <th>low</th>\n",
       "      <th>close</th>\n",
       "      <th>adjclose</th>\n",
       "      <th>volume</th>\n",
       "      <th>ticker</th>\n",
       "      <th>RSIadjclose15</th>\n",
       "      <th>RSIvolume15</th>\n",
       "      <th>...</th>\n",
       "      <th>high-15</th>\n",
       "      <th>K-15</th>\n",
       "      <th>D-15</th>\n",
       "      <th>stochastic-k-15</th>\n",
       "      <th>stochastic-d-15</th>\n",
       "      <th>stochastic-kd-15</th>\n",
       "      <th>volumenrelativo</th>\n",
       "      <th>diff</th>\n",
       "      <th>INCREMENTO</th>\n",
       "      <th>TARGET</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7776</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7777</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7778</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7779</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7780</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>7781 rows × 1285 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       date   open   high    low  close  adjclose  volume  ticker  \\\n",
       "0     False  False  False  False  False     False   False   False   \n",
       "1     False  False  False  False  False     False   False   False   \n",
       "2     False  False  False  False  False     False   False   False   \n",
       "3     False  False  False  False  False     False   False   False   \n",
       "4     False  False  False  False  False     False   False   False   \n",
       "...     ...    ...    ...    ...    ...       ...     ...     ...   \n",
       "7776  False  False  False  False  False     False   False   False   \n",
       "7777  False  False  False  False  False     False   False   False   \n",
       "7778  False  False  False  False  False     False   False   False   \n",
       "7779  False  False  False  False  False     False   False   False   \n",
       "7780  False  False  False  False  False     False   False   False   \n",
       "\n",
       "      RSIadjclose15  RSIvolume15  ...  high-15   K-15   D-15  stochastic-k-15  \\\n",
       "0              True         True  ...     True   True   True             True   \n",
       "1              True         True  ...     True   True   True             True   \n",
       "2              True         True  ...     True   True   True             True   \n",
       "3              True         True  ...     True   True   True             True   \n",
       "4              True         True  ...     True   True   True             True   \n",
       "...             ...          ...  ...      ...    ...    ...              ...   \n",
       "7776          False        False  ...    False  False  False            False   \n",
       "7777          False        False  ...    False  False  False            False   \n",
       "7778          False        False  ...    False  False  False            False   \n",
       "7779          False        False  ...    False  False  False            False   \n",
       "7780          False        False  ...    False  False  False            False   \n",
       "\n",
       "      stochastic-d-15  stochastic-kd-15  volumenrelativo   diff  INCREMENTO  \\\n",
       "0                True              True            False  False       False   \n",
       "1                True              True            False  False       False   \n",
       "2                True              True            False  False       False   \n",
       "3                True              True            False  False       False   \n",
       "4                True              True            False  False       False   \n",
       "...               ...               ...              ...    ...         ...   \n",
       "7776            False             False            False   True        True   \n",
       "7777            False             False            False   True        True   \n",
       "7778            False             False            False   True        True   \n",
       "7779            False             False            False   True        True   \n",
       "7780            False             False            False   True        True   \n",
       "\n",
       "      TARGET  \n",
       "0      False  \n",
       "1      False  \n",
       "2      False  \n",
       "3      False  \n",
       "4      False  \n",
       "...      ...  \n",
       "7776   False  \n",
       "7777   False  \n",
       "7778   False  \n",
       "7779   False  \n",
       "7780   False  \n",
       "\n",
       "[7781 rows x 1285 columns]"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "cd6bffe0-0dfb-4550-a550-c273e672c8e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method DataFrame.corr of             date       open       high        low      close   adjclose  \\\n",
       "0     2022-01-03  17.799999  18.219000  17.500000  17.760000  17.760000   \n",
       "1     2022-01-04  17.700001  18.309999  17.620001  17.660000  17.660000   \n",
       "2     2022-01-05  17.580000  17.799999  16.910000  16.950001  16.950001   \n",
       "3     2022-01-06  16.650000  16.879999  16.139999  16.170000  16.170000   \n",
       "4     2022-01-07  16.219999  16.290001  15.630000  15.710000  15.710000   \n",
       "...          ...        ...        ...        ...        ...        ...   \n",
       "7776  2022-12-23  23.250000  23.540001  23.250000  23.290001  22.699928   \n",
       "7777  2022-12-27  23.350000  23.610001  23.250000  23.350000  22.758406   \n",
       "7778  2022-12-28  23.450001  23.570000  23.219999  23.350000  22.758406   \n",
       "7779  2022-12-29  23.330000  23.740000  23.330000  23.610001  23.011820   \n",
       "7780  2022-12-30  23.680000  23.760000  23.610001  23.610001  23.011820   \n",
       "\n",
       "      volume ticker  RSIadjclose15  RSIvolume15  ...    high-15       K-15  \\\n",
       "0     106600   ASLE            NaN          NaN  ...        NaN        NaN   \n",
       "1     128700   ASLE            NaN          NaN  ...        NaN        NaN   \n",
       "2     103100   ASLE            NaN          NaN  ...        NaN        NaN   \n",
       "3     173600   ASLE            NaN          NaN  ...        NaN        NaN   \n",
       "4     137800   ASLE            NaN          NaN  ...        NaN        NaN   \n",
       "...      ...    ...            ...          ...  ...        ...        ...   \n",
       "7776    4900   ATLO      60.782255    47.081752  ...  23.600000  26.223672   \n",
       "7777    9200   ATLO      62.022801    47.747952  ...  23.610001  30.764722   \n",
       "7778   15200   ATLO      62.022801    48.713225  ...  23.610001  30.764722   \n",
       "7779    7100   ATLO      67.186408    47.445460  ...  23.740000  46.457382   \n",
       "7780    7100   ATLO      67.186408    47.445460  ...  23.760000  45.784072   \n",
       "\n",
       "           D-15  stochastic-k-15  stochastic-d-15  stochastic-kd-15  \\\n",
       "0           NaN              NaN              NaN               NaN   \n",
       "1           NaN              NaN              NaN               NaN   \n",
       "2           NaN              NaN              NaN               NaN   \n",
       "3           NaN              NaN              NaN               NaN   \n",
       "4           NaN              NaN              NaN               NaN   \n",
       "...         ...              ...              ...               ...   \n",
       "7776  27.022465        26.223672        27.022465         -0.798793   \n",
       "7777  28.003602        30.764722        28.003602          2.761119   \n",
       "7778  29.251039        30.764722        29.251039          1.513683   \n",
       "7779  35.995609        46.457382        35.995609         10.461773   \n",
       "7780  41.002059        45.784072        41.002059          4.782013   \n",
       "\n",
       "      volumenrelativo      diff  INCREMENTO  TARGET  \n",
       "0            0.919758 -1.900001   -9.664295       0  \n",
       "1            1.110440 -1.379999   -7.247895       0  \n",
       "2            0.889560 -0.930000   -5.201344       0  \n",
       "3            1.497843 -0.360000   -2.177856       0  \n",
       "4            1.188956 -0.120000   -0.758054       0  \n",
       "...               ...       ...         ...     ...  \n",
       "7776         0.333333       NaN         NaN       0  \n",
       "7777         0.625850       NaN         NaN       0  \n",
       "7778         1.034014       NaN         NaN       0  \n",
       "7779         0.482993       NaN         NaN       0  \n",
       "7780         0.482993       NaN         NaN       0  \n",
       "\n",
       "[7781 rows x 1285 columns]>"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "666dea62-4c1c-4641-a64d-ca98667b18de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0\n",
       "1    0\n",
       "2    0\n",
       "3    0\n",
       "4    0\n",
       "Name: TARGET, dtype: int64"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feature=df.iloc[:,1:5]\n",
    "feature.head()\n",
    "target=df.TARGET\n",
    "target.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d83e5111-dff0-4431-aa75-7e6a675c16f8",
   "metadata": {},
   "source": [
    "# Checking if any character or string type entries are there or not"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "8b4e4a25-a233-4ae7-a736-3f54fb29b50d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       17.799999\n",
       "1       17.700001\n",
       "2       17.580000\n",
       "3       16.650000\n",
       "4       16.219999\n",
       "          ...    \n",
       "7776    23.250000\n",
       "7777    23.350000\n",
       "7778    23.450001\n",
       "7779    23.330000\n",
       "7780    23.680000\n",
       "Name: open, Length: 7781, dtype: float64"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.to_numeric(df.open,errors='coerce')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad129775-dbe1-4dbc-9871-2a43be36571f",
   "metadata": {},
   "source": [
    "# Creating Correlation matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "645ac485-5a5e-4211-ad98-ccd4eec284a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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nvz8/OTp6amwsLAip9v++c9/VufOnTVhwgSdP39erVq1sq2y27RpUw0cOPCW3u+NdOrUSZ06dbphTGRkpMLDw/X000/LMAxVqlRJH3/8sZKSkgrFNmrUSJI0a9YsPfLII7JarYqIiPjNIv+37Nq1S08++aQeeeQRWxG6YMEC/d///Z9mzpyp2NjYEl0PAABXRIcUAEpg2LBhSk1NVVRUlF566SV16tRJvXv31vvvv68BAwZo3rx5tti5c+fqxRdf1Nq1a9W9e3dNnDhRnTp1UkpKSpFFXUl06NBB27ZtU4UKFRQbG6uOHTtqxIgR+uyzz9SxY8cbnvvee++pQ4cOGj9+vPr06aPU1FQlJSUpICDALi4sLEwzZ87UN998o/bt26tFixa2DvCvWSwWrVq1SqNHj9Y///lPde3aVa+++qoGDhyozz//vMh9W39PVqtVH3/8se6880499thjeuihh3Ty5El99tlnhWLbt2+vuLg4ffzxx2rdurVatGhht5docVy8eFF9+/ZVWFiY5syZYxt/4IEHNGrUKI0fP17btm275fcFAICzsxiGYZidBAAAAADg9kOHFAAAAABgCgpSAAAAAIApKEgBAAAAAKagIAUAAACA39mXX36pHj16qHr16rbFAW8mOTlZUVFR8vT01B133KG33nqrUMyKFStUv359eXh4qH79+lq5cmWhmDlz5igsLEyenp6KiorSpk2b7J43DEOTJ09W9erV5eXlpfbt2+u7776zi8nNzdUTTzyhwMBA+fj4qGfPnjp27FjJPgRRkAIAAADA7+7ixYtq3Lix3njjjWLFHzp0SF27dlWbNm20c+dOPfPMM3ryySe1YsUKW8yWLVvUr18/DRw4UN98840GDhyovn37auvWrbaY5cuXKzY2VhMnTtTOnTvVpk0bdenSRenp6baYl19+WQkJCXrjjTe0fft2VatWTX/+85914cIFW0xsbKxWrlypZcuWafPmzcrOzlb37t0L7Wt+M6yyCwAAAAAmslgsWrlypXr37v2bMRMmTNCaNWu0d+9e29jw4cP1zTffaMuWLZKkfv366fz58/rPf/5ji7nvvvtUsWJFvf/++5Kkli1bqlmzZpo7d64tpl69eurdu7emT58uwzBUvXp1xcbGasKECZKudUODgoL00ksv6bHHHtO5c+dUpUoVJSYmql+/fpKk48eP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UlLUtm1beXl5qWbNmnryySd18eJF2/O1a9fWCy+8oEcffVR+fn6qVauW5s2b97u+T+C6goICjR8/XpUqVVK1atU0efJk23O/nrKbkpKiJk2ayNPTU82bN9eqVauKnBKWlpam5s2by9vbWzExMdq3b9/v82aAEsjKytJf//pXVaxYUd7e3urSpYsOHDggSTIMQ1WqVNGKFSts8U2aNFHVqlVtj7ds2SKr1ars7OzfPXfguoKCAr300kuqU6eOPDw8VKtWLU2bNq3I2OTkZN19993y8PBQcHCwnn76aV29etX2/IcffqhGjRrJy8tLlStXVseOHe3+/8s///lP1atXT56enoqMjNScOXPK/P0BcAwFqRMbP368VqxYocWLF2vHjh2qU6eOOnfurDNnzthixo0bp1dffVXbt29X1apV1bNnT+Xl5UmSdu3apc6dO6tPnz769ttvtXz5cm3evFmPP/643evMmDFDzZs3186dOzVy5EiNGDFC33///e/6XgFJWrx4sXx8fLR161a9/PLLmjp1qpKSkgrFXbhwQT169FCjRo20Y8cOPffcc5owYUKR15w4caJmzJih1NRUlS9fXo8++mhZvw2gxAYNGqTU1FStWbNGW7ZskWEY6tq1q/Ly8mSxWNS2bVtt3LhR0rXidc+ePcrLy9OePXskXftHyqioKPn6+pr4LnC7i4uL00svvaT4+Hjt2bNH7733noKCggrFZWRkqGvXrmrRooW++eYbzZ07VwsWLNDzzz8v6dotGg899JAeffRR7d27Vxs3blSfPn1kGIYkaf78+Zo4caKmTZumvXv36oUXXlB8fLwWL178u75fAMVkwCllZ2cbVqvVePfdd21jV65cMapXr268/PLLxhdffGFIMpYtW2Z7/vTp04aXl5exfPlywzAMY+DAgcbf/vY3u+tu2rTJKFeunHHp0iXDMAwjNDTU+Mtf/mJ7vqCgwKhataoxd+7csnx7QCHt2rUzWrdubTfWokULY8KECYZhGIYkY+XKlYZhGMbcuXONypUr237HhmEY8+fPNyQZO3fuNAzDsP038tlnn9liPvnkE0OS3XmAWdq1a2c89dRTxv79+w1Jxn//+1/bc6dOnTK8vLyMf/3rX4ZhGMbs2bONhg0bGoZhGKtWrTKaN29u9OnTx3jzzTcNwzCMTp062f5bAcxw/vx5w8PDw5g/f36h5w4dOmT39/MzzzxjREREGAUFBbaYN9980/D19TXy8/ONtLQ0Q5Jx+PDhIl+rZs2axnvvvWc39txzzxnR0dGl94YAlBo6pE7qhx9+UF5enlq1amUbs1qtuvvuu7V3717bWHR0tO3PlSpVUkREhO35tLQ0LVq0SL6+vrajc+fOKigo0KFDh2zn3XXXXbY/WywWVatWTSdPnizLtwcU6Ze/RUkKDg4u8re4b98+3XXXXfL09LSN3X333Te9ZnBwsCTx+8Yfyt69e1W+fHm1bNnSNla5cmW7v8/bt2+v7777TqdOnVJycrLat2+v9u3bKzk5WVevXlVKSoratWtn1lsAtHfvXuXm5uree+8tVmx0dLQsFottrFWrVsrOztaxY8fUuHFj3XvvvWrUqJEefPBBzZ8/X1lZWZKkn3/+WUePHtWQIUPs/v/N888/rx9++KHM3h8Ax7GokZMy/v+0lF/+ZX19/Ndjv3b9+YKCAj322GN68sknC8XUqlXL9mer1Vro/IKCAofyBm5FcX+LRf13cP2/mRtd85f/bQB/FL/12/3l77xhw4aqXLmykpOTlZycrKlTp6pmzZqaNm2atm/frkuXLql169a/Z9qAHS8vr2LH3ujvcIvFIjc3NyUlJSklJUWffvqpXn/9dU2cOFFbt26Vt7e3pGvTdn/5jziS5ObmdovvAkBZoEPqpOrUqSN3d3dt3rzZNpaXl6fU1FTVq1fPNvbVV1/Z/pyVlaX9+/crMjJSktSsWTN99913qlOnTqHD3d3993szQCmLjIzUt99+q9zcXNtYamqqiRkBjqtfv76uXr2qrVu32sZOnz6t/fv32/6+v34f6erVq7V79261adNGjRo1Ul5ent566y01a9ZMfn5+Zr0FQHXr1pWXl5c2bNhw09j69esrJSXF7h9jUlJS5Ofnp5CQEEnXfvOtWrXSlClTtHPnTrm7u2vlypUKCgpSSEiIfvzxx0L/3yYsLKzM3h8Ax1GQOikfHx+NGDFC48aN07p167Rnzx4NGzZMOTk5GjJkiC1u6tSp2rBhg3bv3q1BgwYpMDBQvXv3liRNmDBBW7Zs0ahRo/T111/rwIEDWrNmjZ544gmT3hVQOgYMGKCCggL97W9/0969e7V+/Xq9+uqrkgrPKgD+6OrWratevXpp2LBh2rx5s7755hv95S9/UUhIiHr16mWLa9++vd577z3ddddd8vf3txWp7777rtq3b2/eGwAkeXp6asKECRo/fryWLFmiH374QV999ZUWLFhQKHbkyJE6evSonnjiCX3//fdavXq1nn32WY0ePVrlypXT1q1b9cILLyg1NVXp6en66KOP9PPPP9v+gWby5MmaPn26Zs2apf3792vXrl365z//qYSEhN/7bQMoBqbsOrEXX3xRBQUFGjhwoC5cuKDmzZtr/fr1qlixol3MU089pQMHDqhx48Zas2aNrft51113KTk5WRMnTlSbNm1kGIbCw8PVr18/s94SUCr8/f318ccfa8SIEWrSpIkaNWqkf/zjHxowYIDdfaWAs/jnP/+pp556St27d9eVK1fUtm1brV271m7KeYcOHZSfn29XfLZr106rVq3i/lH8IcTHx6t8+fL6xz/+oePHjys4OFjDhw8vFBcSEqK1a9dq3Lhxaty4sSpVqqQhQ4Zo0qRJkq79Hf/ll19q5syZOn/+vEJDQzVjxgx16dJFkjR06FB5e3vrlVde0fjx4+Xj46NGjRopNjb293y7AIrJYvzWzSlwahs3blSHDh2UlZWlChUqmJ0OYLp3331XgwcP1rlz50p0LxMAAADKDh1SAC5pyZIluuOOOxQSEqJvvvlGEyZMUN++fSlGAQAA/kAoSAG4pBMnTugf//iHTpw4oeDgYD344IOaNm2a2WkBAADgF5iyCwAAAAAwBavsAgAAAABMQUEKAAAAADAFBSkAAAAAwBQUpAAAAAAAU1CQAgAAAABMQUEKAAAAADAFBSkAAAAAwBQUpAAAAAAAU1CQAgAAAABM8f8A2zRpj14hk0gAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1200x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12, 8))\n",
    "g=sns.heatmap(feature.corr(),annot=True, cmap='coolwarm', linewidths=0.5)\n",
    "plt.title('Correlation Matrix')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa88404e-0556-4cb5-87cf-4af5c63a2f8a",
   "metadata": {},
   "source": [
    "# Checking the weight of selected features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "14446d53-8764-4a01-8ed5-08f11166452d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.23970509 0.25658095 0.2495694  0.25414456]\n"
     ]
    }
   ],
   "source": [
    "model = ExtraTreesClassifier()\n",
    "model.fit(feature,target)\n",
    "print(model.feature_importances_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4da51ff8-f0ef-4dc4-80cc-19b024e557a9",
   "metadata": {},
   "source": [
    "# Graphical representation of weight"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "309b015f-053c-42bb-a218-a5ea1b5fe749",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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AEkSMQBAkkQMAJAkEQMAJEnEAABJEjEAQJJEDACQJBEDACSpqLUn0NwOnb4g2pV0au1pAMAua+2s01p7CjvFOzEAQJJEDACQJBEDACRJxAAASRIxAECSRAwAkCQRAwAkScQAAEkSMQBAkkQMAJAkEQMAJEnEAABJEjEAQJJEDACQJBEDACRJxAAASRIxAECSmjViRo8eHeXl5c35EgDAbso7MQBAkkQMAJCkFouYDRs2xDnnnBN77rlndOrUKU455ZRYs2ZNRERkWRb77LNPPPzww4XxQ4cOje7duxd+XrJkSRQXF8cnn3zSUlMGAHZhLRYx48ePj2XLlsX8+fNjyZIlkWVZnHrqqbF58+bI5XJx/PHHx8KFCyNia/CsWrUqNm/eHKtWrYqIiIULF8awYcNijz32qPf41dXVUVVVVWsDANquFomYNWvWxPz58+N3v/tdHHfccTFkyJCYN29evP322/Hoo49GxNabgLdFzNNPPx1DhgyJr371q4V9CxcujNGjRzf4Gtddd1107dq1sJWVlTXvSQEArapFImb16tVRVFQURx11VGFft27dol+/frF69eqI2Boxr7zySnzwwQexaNGiGD16dIwePToWLVoUW7ZsiYqKihg1alSDrzFt2rSorKwsbOvWrWv28wIAWk+LREyWZQ3uz+VyERFx6KGHRrdu3WLRokWFiBk1alQsWrQonnvuufjPf/4Txx57bIOvUVJSEvl8vtYGALRdLRIxAwcOjC1btsTSpUsL+z788MN4/fXXY8CAARERhftiHnvssVi5cmUcd9xxMWjQoNi8eXPcfvvtcfjhh0eXLl1aYroAQAJaJGIOOeSQOOOMM2LixImxePHiePHFF+P73/9+9OrVK84444zCuNGjR8e9994bgwcPjnw+XwibefPmbfd+GABg99Ni30666667YtiwYXH66afHiBEjIsuyePzxx6O4uLgw5oQTTojPPvusVrCMGjUqPvvss+3eDwMA7H5yWUM3rCSuqqpq67eUyv8Q7Uo6tfZ0AGCXtXbWaa09hYJtv78rKyu/8P5W/2IvAJAkEQMAJEnEAABJEjEAQJJEDACQJBEDACRJxAAASRIxAECSRAwAkCQRAwAkScQAAEkSMQBAkkQMAJAkEQMAJEnEAABJEjEAQJKKWnsCzW3lzDGRz+dbexoAQBPzTgwAkCQRAwAk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r/fRRx8dRx99dOHxkSNHxuGHHx4333xz/PrXv26qaSerOa5F13fDmnptXN/bt7Prfd9998WMGTPisccei+7duzfJMXcHTb3eu+r1nUTE7L333tG+ffs6Ffnee+/Vqc1tevToUe/4oqKi6Nat23bHNHTM3UVzrffntWvXLo444ohWL/nWtjPrvSNc3/VrrvX+PNf3Vl9mvR944IGYMGFCPPjgg3HSSSfVesz1Xb/mWu/P21Wu7yQ+TurQoUMMGzYsnnzyyVr7n3zyyTjmmGPqfc6IESPqjP/rX/8aw4cPj+Li4u2OaeiYu4vmWu/Py7IsVqxYET179myaiSdqZ9Z7R7i+69dc6/15ru+tdna977vvvhg/fnzce++9cdppp9V53PVdv+Za78/bZa7v1ribeGds+8rYnXfema1atSorLy/POnfunK1duzbLsiy74oorsrPPPrswfttXfn/84x9nq1atyu688846X/n929/+lrVv3z6bNWtWtnr16mzWrFm+ovf/mmO9Z8yYkT3xxBPZG2+8kS1fvjw777zzsqKiomzp0qUtfn67msaud5Zl2fLly7Ply5dnw4YNy8aNG5ctX748e+WVVwqPu74b1hzr7fpuWGPX+957782KioqyW2+9tdbXeT/++OPCGNd3w5pjvXfV6zuZiMmyLLv11luz3r17Zx06dMgOP/zwbNGiRYXHzj333GzUqFG1xi9cuDA77LDDsg4dOmR9+vTJfvOb39Q55oMPPpj169cvKy4uzvr37589/PDDzX0ayWjq9S4vL88OOOCArEOHDtk+++yTnXzyyVlFRUVLnEoSGrveEVFn6927d60xru+GNfV6u763rzHrPWrUqHrX+9xzz611TNd3w5p6vXfV6zuXZf9/9yUAQEKSuCcGAODzRAwAkCQRAwAkScQAAEkSMQBAkkQMAJAkEQMAJEnEAABJEjEAQJJEDACQJBEDACRJxAAASfo/WCYfeEb57ZoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "feat_importances = pd.Series(model.feature_importances_, index=feature.columns)\n",
    "feat_importances.nlargest(10).plot(kind='barh')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62a29832-1716-4825-996e-594c76ef7c1d",
   "metadata": {},
   "source": [
    "# Ploting box-plot for oberving outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "299b7e6e-d640-423f-94ff-91b0862f05ea",
   "metadata": {},
   "source": [
    "- > By observing the box-plot,we can that in the selected feature there are most of data is in the category of outlier.But as i am using RandomForestClassifier,outliers can easily be handled."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "95c16eb8-bee5-46bd-98fe-60953737a5e5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYEAAAGHCAYAAABWAO45AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABANUlEQVR4nO3de1xUdf4/8NdwG+6jgDKOXKQkK0EzFJS2hQRpzetaaWqmG+1WlknqUq7fivxukLYqJY9sVX5KGuLWht1WBdukXEIRNW+t1opclAlNGEBuCp/fH34528gAB2acgZnX8/GYx0PO+8PM5yDMa875nPP5KIQQAkREZJPsLN0BIiKyHIYAEZENYwgQEdkwhgARkQ1jCBAR2TCGABGRDWMIEBHZMIYAEZENYwgQEdkwhgD1Svv374dCocD+/fst3RWL+f7777FgwQIEBATAyckJPj4+eOihh7B7926jnvfdd9/F1q1bTdPJLtTX1yMpKcmm/x97O4YAUS/08ccfY9SoUTh06BBeeeUV7Nu3Dxs2bAAAPPTQQ0hMTOzxc5s7BF5//XWGQC/mYOkOEJG+//znP5g3bx5CQ0Oxf/9+uLm5SbVHH30Uzz77LN566y3ce++9eOyxxyzYU7IGPBIgi/j3v/+N2bNnw9fXF0qlEgEBAXjiiSfQ1NTU6fd9+umnGDduHFxdXeHh4YEJEybg22+/1Wtz6dIl/OEPf4C/vz+USiUGDBiA++67D/v27dNrt2/fPsTExMDT0xOurq6477778OWXX3b6+pcuXYKTkxNeeeUVg/ukUCjwzjvvALjxKXjZsmUICgqCs7MzvLy8MHr0aOzYsaPT11i3bh3q6+uxfv16vQBos2bNGvTr1w9vvPGGtC0pKQkKhaJd261bt0KhUOD8+fMAgCFDhuDUqVPIy8uDQqGAQqHAkCFDAPz3FNz27duxZMkSqNVquLi4ICoqCkePHtV73ujoaERHR7d7vQULFkjPd/78eQwYMAAA8Prrr0uvt2DBgk73n8yLIUBm991332HMmDEoKCjAypUrsXv3bqSkpKCpqQnNzc0dfl9mZiamTZsGT09P7NixA+np6aiqqkJ0dDQOHDggtZs3bx527dqFV199FTk5Odi8eTNiY2Px888/S222b9+OuLg4eHp6IiMjA3/729/g5eWFBx98sNMgGDBgACZPnoyMjAy0trbq1bZs2QInJyfMnTsXALBkyRJs2LABL7zwAvbs2YNt27bh0Ucf1euHIbm5ufD19cXYsWMN1l1dXREXF4eTJ09Cq9V2+lw3y87Oxm233YZRo0bh22+/xbfffovs7Gy9Nn/6059w7tw5bN68GZs3b8bFixcRHR2Nc+fOdeu1Bg0ahD179gAA4uPjpdczFKBkQYLIzMaPHy/69esnKisrO2zz1VdfCQDiq6++EkII0dLSIjQajQgNDRUtLS1Su9raWjFw4EARGRkpbXN3dxcJCQkdPvfVq1eFl5eXmDJlit72lpYWMXLkSBEeHt5p/z/99FMBQOTk5Ejbrl+/LjQajXj44YelbSEhIWL69OmdPpchzs7OYuzYsZ22eemllwQAcfDgQSGEEK+99pow9Oe8ZcsWAUAUFxdL24YPHy6ioqLatW37md97772itbVV2n7+/Hnh6OgonnrqKWlbVFSUweeYP3++CAwMlL6+dOmSACBee+21TveHLIdHAmRW9fX1yMvLw8yZM6VTBXKcOXMGFy9exLx582Bn999fW3d3dzz88MMoKChAfX09ACA8PBxbt27Fn//8ZxQUFODatWt6z5Wfn48rV65g/vz5uH79uvRobW3Fb37zGxQWFuLq1asd9mXixIlQq9XYsmWLtG3v3r24ePEinnzySWlbeHg4du/ejZdffhn79+9HQ0OD7P3tivi/ZUAMnQIy1pw5c/SeNzAwEJGRkfjqq69M/lpkeQwBMquqqiq0tLTAz8+vW9/Xdgpl0KBB7WoajQatra2oqqoCAOzcuRPz58/H5s2bMW7cOHh5eeGJJ56QTp389NNPAIBHHnkEjo6Oeo9Vq1ZBCIErV6502BcHBwfMmzcP2dnZqK6uBnDj3PugQYPw4IMPSu3eeecdvPTSS9i1axceeOABeHl5Yfr06fjhhx863deAgAAUFxd32qbtHL+/v3+n7XpCrVYb3NbVaSzqmxgCZFZeXl6wt7dHeXl5t77P29sbAFBRUdGudvHiRdjZ2aF///4AAB8fH6SmpuL8+fMoKSlBSkoKPv74Y2lA0sfHBwCwfv16FBYWGnz4+vp22p/f/e53aGxsRFZWFqqqqvDpp5/iiSeegL29vdTGzc0Nr7/+Ov79739Dq9Viw4YNKCgowJQpUzp97gkTJuCnn35CQUGBwXp9fT1yc3MREhIivWE7OzsDQLuB9cuXL3f6WoYYGmfQarXS/0Hb6xkaxO/J65GFWfp8FNme8ePHi/79+4tLly512MbQmMDgwYPFPffco3e+uq6uTgwcOFDcd999nb7m9OnTxYABA4QQN8YR+vXrJ5599lmj9iMiIkKEh4eLtLQ0AUD8+9//7vJ7EhISBABx9erVDtv8+OOPwsXFRYwePVrU1dW1qz/77LMCgMjKypK27dixQwAQhw4d0mv761//ut2YwL333mtw3KPtZx4WFmZwTCA+Pl7a9vTTTwsvLy/R2Ngobbt8+bLo37+/3phATU2NACASExM73F+yLIYAmd2xY8eEu7u7uO2228TGjRvFP//5T7Fjxw4xe/ZsUVNTI4RoHwJCCPHBBx8IAOKhhx4Sn3zyifjb3/4mxowZI5ycnMQ333wjhBCiurpajBo1Srz11lvis88+E/v37xdvvfWWcHZ2FnPmzJGea9u2bcLOzk7MmjVLfPjhhyIvL0989NFH4pVXXhHPPPOMrP3461//KgAIPz8/vYHpNuHh4WLlypVi165dIi8vT7z33nvC29tbjBs3rsvn/uijj4RSqRR33XWX2LRpk/j666/Fhx9+KCZOnCgAiGXLlum11+l0wsvLS4SGhors7Gzx2WefiYcfflgEBQW1C4H58+cLpVIpsrKyxKFDh8Tx48f1fub+/v5i2rRp4vPPPxcffPCBGDp0qPDw8BA//vij9BwHDhwQAMQjjzwi9u7dKzIzM8U999wjAgMD9UJACCECAwPFsGHDxN69e0VhYaFeX8jyGAJkEadPnxaPPvqo8Pb2Fk5OTiIgIEAsWLBA+mRpKASEEGLXrl0iIiJCODs7Czc3NxETEyP+9a9/SfXGxkbxzDPPiBEjRghPT0/h4uIihg0bJl577bV2n77z8vLEpEmThJeXl3B0dBSDBw8WkyZNEh9++KGsfdDpdMLFxUUAEJs2bWpXf/nll8Xo0aNF//79hVKpFLfddpt48cUXxeXLl2U9/6lTp8T8+fOFn5+fcHR0FF5eXuI3v/mN+OKLLwy2P3TokIiMjBRubm5i8ODB4rXXXhObN29uFwLnz58XcXFxwsPDQwCQ3rTbfubbtm0TL7zwghgwYIBQKpXi/vvvF4cPH273ehkZGeKuu+4Szs7O4u677xY7d+5sd3WQEELs27dPjBo1SiiVSgFAzJ8/X9b+k3kohPi/ywyIyKbt378fDzzwAD788EM88sgjlu4OmQkHhomIbBhDgIjIhvF0EBGRDeORABGRDWMIEBHZMIYAEZEN46IyAFpbW3Hx4kV4eHjckgm5iIjMTQiB2tpaaDQavUkXb8YQwI25Z27FRFxERJZWVlbW6YSNDAEAHh4eAG78sDw9PS3cGyIi49XU1MDf3196f+sIQwD/nZPd09OTIUBEVqWrU9wcGCYismEMASIiG8YQICKyYQwBIiIbxhAgIrJhDAEiIhtm0RC4fv06/ud//gdBQUFwcXHBbbfdhpUrV6K1tVVqI4RAUlISNBoNXFxcEB0djVOnTuk9T1NTExYtWgQfHx+4ublh6tSp3V7InLonPz8fs2bNQn5+vqW7QkRGsGgIrFq1Cu+99x7S0tLw/fffY/Xq1Xjrrbewfv16qc3q1auxdu1apKWlobCwEGq1GhMmTEBtba3UJiEhAdnZ2cjKysKBAwdQV1eHyZMno6WlxRK7ZfUaGxuxdu1a/PTTT1i7di0aGxst3SUi6ilLrm05adIk8eSTT+ptmzFjhnj88ceFEEK0trYKtVot3nzzTane2NgoVCqVeO+994QQNxYWd3R0FFlZWVKbCxcuCDs7O7Fnzx5Z/dDpdAKA0Ol0xu6STdi8ebOIjo4WUVFRIjo6WqSnp1u6S0R0E7nvaxY9EvjVr36FL7/8EmfPngUAfPfddzhw4AAeeughAEBxcTG0Wi3i4uKk71EqlYiKipJOQxQVFeHatWt6bTQaDUJCQjo8VdHU1ISamhq9B8lTXl6OzMxMiP9bi0gIgczMTJ5+I+qjLBoCL730EmbPno0777wTjo6OGDVqFBISEjB79mwAgFarBQD4+vrqfZ+vr69U02q1cHJyQv/+/Ttsc7OUlBSoVCrpwcnj5BFC4O233+5wu+AidUR9jkVDYOfOndi+fTsyMzNx5MgRZGRk4C9/+QsyMjL02t0894UQosv5MDprs3z5cuh0OulRVlZm3I7YiNLSUhQWFrYba2lpaUFhYSFKS0st1DMi6imLTiD3xz/+ES+//DIee+wxAEBoaChKSkqQkpKC+fPnQ61WA7jxaX/QoEHS91VWVkpHB2q1Gs3NzaiqqtI7GqisrERkZKTB11UqlVAqlbdqt6xWQEAAxowZgyNHjugFgb29PcLCwhAQEGDB3hFRT1j0SKC+vr7dYgf29vbSJaJBQUFQq9XIzc2V6s3NzcjLy5Pe4MPCwuDo6KjXpqKiAidPnuwwBKhnFAoFFi9ebPC0z+LFi7kgD1EfZNEjgSlTpuCNN95AQEAAhg8fjqNHj2Lt2rV48sknAdx400lISEBycjKCg4MRHByM5ORkuLq6Ys6cOQAAlUqF+Ph4LF26FN7e3vDy8sKyZcsQGhqK2NhYS+6eVfLz88Pw4cNx4sQJadvw4cMxePBgC/aKiHrKoiGwfv16vPLKK1i4cCEqKyuh0Wjw9NNP49VXX5XaJCYmoqGhAQsXLkRVVRUiIiKQk5Ojt1DCunXr4ODggJkzZ6KhoQExMTHYunUr7O3tLbFbVq28vBynT5/W23b69GmUl5d3unoREfVOCsFLOlBTUwOVSgWdTsdFZTohhEBiYqLBMYF7770Xq1ev5ikhol5C7vsa5w4i2Xh1EJH1YQiQbG1XB918ms3e3h7h4eG8OoioD2IIkGy8OojI+jAEqFvarg76JV4dRNR3MQSoW8rLy9tN5X3q1CnOHUTURzEESLaO5ghqbW3l3EFEfRRDgGRruzro5jd7IQSvDiLqoxgCJJu/v3+H1xt7enpyNlaiPoghQLKVlZV1uPZCTU0NZ2Ml6oMYAiRb230ChvA+AaK+iSFAsikUCsTExBisxcTE8D4Boj6IIUCytba2Ii0tzWBt/fr10hTgRNR3MARItoKCAtTV1Rms1dXVoaCgwMw9IiJjMQRItl+u7taTOhH1PgwBkm3IkCG44447DNaGDRuGIUOGmLdDRGQ0hgDJplAo8Ic//MFg7Q9/+AMHhon6IIYAySaEwMaNGw3W/vrXv3LaCKI+iCFAsp0/fx5nz541WDt79izOnz9v3g4RkdEYAiRbRUWFUXUi6n0YAiTb2LFj4e7ubrDm7u6OsWPHmrlHRGQshgDJZmdnh6SkJIO1lStXws6Ov05EfQ3/aqlb1Gq1we0DBw40c0+IyBQYAiRb26IyhnBRGaK+iSFAsrUtKmMIF5Uh6pssGgJDhgyBQqFo93juuecA3PjkmZSUBI1GAxcXF0RHR7db37apqQmLFi2Cj48P3NzcMHXqVK53e4v4+/t3OjDMRWWI+h6LhkBhYSEqKiqkR25uLgDg0UcfBQCsXr0aa9euRVpaGgoLC6FWqzFhwgTU1tZKz5GQkIDs7GxkZWXhwIEDqKurw+TJk9HS0mKRfbJmpaWlnU4gxyMBor7HoiEwYMAAqNVq6fH555/j9ttvR1RUFIQQSE1NxYoVKzBjxgyEhIQgIyMD9fX1yMzMBADodDqkp6djzZo1iI2NxahRo7B9+3acOHEC+/bts+SuERH1Cb1mTKC5uRnbt2/Hk08+CYVCgeLiYmi1WsTFxUltlEoloqKikJ+fDwAoKirCtWvX9NpoNBqEhIRIbQxpampCTU2N3oO6FhAQ0OnpIK4sRtT39JoQ2LVrF6qrq7FgwQIAgFarBQD4+vrqtfP19ZVqWq0WTk5O6N+/f4dtDElJSYFKpZIePJctT1lZWaeng7jGMFHf02tCID09HRMnToRGo9HbfvPMlEKILmer7KrN8uXLodPppAffvOTx8/ODvb29wZq9vT38/PzM3CMiMlavCIGSkhLs27cPTz31lLSt7aakmz/RV1ZWSkcHarUazc3NqKqq6rCNIUqlEp6ennoP6trBgwc7HHBvaWnBwYMHzdwjIjJWrwiBLVu2YODAgZg0aZK0LSgoCGq1WrpiCLgxbpCXl4fIyEgAQFhYGBwdHfXaVFRU4OTJk1IbMp2u7grmXcNEfY+DpTvQ2tqKLVu2YP78+XBw+G93FAoFEhISkJycjODgYAQHByM5ORmurq6YM2cOAEClUiE+Ph5Lly6Ft7c3vLy8sGzZMoSGhiI2NtZSu2S1jh8/3mV96NChZuoNEZmCxUNg3759KC0txZNPPtmulpiYiIaGBixcuBBVVVWIiIhATk4OPDw8pDbr1q2Dg4MDZs6ciYaGBsTExGDr1q0dnrumnvPx8TGqTkS9j0JwwhfU1NRApVJBp9NxfKATX3/9NV599dUO6ytXrsSvf/1rM/aIiDoi932tV4wJUN9w+fJlo+pE1PswBEi2e+65x6g6EfU+DAGSLSgoqMNLb9VqNYKCgszcIyIyFkOAZBNCoLq62mCtqqqK6wkQ9UEMAZItPz8fTU1NBmtNTU2dztdE1FPp6ekYP3480tPTLd0Vq8QQINkqKyuNqhN1V3V1NT744AO0trbigw8+6PBIlHqOIUCycWCYzO2VV15Ba2srgBs3lnZ2iTL1DEOAZOtqqmhOJU2mdPjwYZw4cUJv2/Hjx3H48GEL9cg6MQRIts8++8yoOpFcra2tWLlypcHaypUrpaMDMh5DgGTz8vIyqk4k18GDBztc7KmmpoYz1poQQ4Bk4x3DZC7h4eGdrl0RHh5u5h5ZL4YAycYJ5MhcysvLO127ory83Mw9sl4MAZLtypUrRtWJ5AoICMCYMWMM1sLDw3kRggkxBEi2KVOmGFUnkkuhUGDx4sXtlom1s7MzuJ16jiFAsl24cMGoOlF3+Pn54fHHH9fb9vjjj2Pw4MEW6pF1YgiQbF398fGPk0xt7ty50ljTgAEDpFUFyXQYAiQb7xMgc3N2dsaSJUvg6+uLF198Ec7OzpbuktWx+PKS1Hf079/fqDpRT0RGRiIyMtLS3bBaPBIg2X788Uej6kTU+zAESLYHHnjAqDoR9T4MAZJtyJAhRtWJqPdhCJBsXc3XwvlciPoehgDJplarjaoTUe/DECDZulpDmGsM062Qn5+PWbNmcfnSW8TiIXDhwgU8/vjj8Pb2hqurK+655x4UFRVJdSEEkpKSoNFo4OLigujoaJw6dUrvOZqamrBo0SL4+PjAzc0NU6dO5QRTt8CxY8eMqhN1V2NjI1JSUvDTTz8hJSUFjY2Nlu6S1bFoCFRVVeG+++6Do6Mjdu/ejdOnT2PNmjXo16+f1Gb16tVYu3Yt0tLSUFhYCLVajQkTJqC2tlZqk5CQgOzsbGRlZeHAgQOoq6vD5MmTO5yFkHqGRwJkbhkZGdLfem1tLd5//30L98j6WPRmsVWrVsHf3x9btmyRtv3yChMhBFJTU7FixQrMmDEDwI1fCl9fX2RmZuLpp5+GTqdDeno6tm3bhtjYWADA9u3b4e/vj3379uHBBx806z5Zs64m7eKkXmRK5eXl2LFjh962zMxMPPTQQ/Dz87NQr6yPRY8EPv30U4wePRqPPvooBg4ciFGjRmHTpk1Svbi4GFqtFnFxcdI2pVKJqKgo6fxgUVERrl27ptdGo9EgJCSkw3OITU1NqKmp0XtQ10aMGGFUnUguIQRWrVplsLZq1SoedZqQRUPg3Llz2LBhA4KDg7F3714888wzeOGFF6RDPq1WCwDw9fXV+z5fX1+pptVq4eTk1G7Kgl+2uVlKSgpUKpX08Pf3N/WuWaXvvvvOqDqRXCUlJe0WmW9z4sQJlJSUmLlH1suiIdDa2op7770XycnJGDVqFJ5++mn8/ve/x4YNG/Ta3XyaQQjR5amHztosX74cOp1OepSVlRm3IzaiuLjYqDqRXBx/Mh+LhsCgQYNw991362276667UFpaCuC/153f/Im+srJSOjpQq9Vobm5GVVVVh21uplQq4enpqfegro0ePdqoOhH1PhYNgfvuuw9nzpzR23b27FkEBgYCAIKCgqBWq5GbmyvVm5ubkZeXJ80qGBYWBkdHR702FRUVOHnyJGceNDEuL0nmwosQzMeiVwe9+OKLiIyMRHJyMmbOnIlDhw5h48aN2LhxI4Ab/9EJCQlITk5GcHAwgoODkZycDFdXV2lxCZVKhfj4eCxduhTe3t7w8vLCsmXLEBoaKl0tRKYxcOBAo+pEcgUEBMDd3R11dXXtau7u7lxj2IQsGgJjxoxBdnY2li9fjpUrVyIoKAipqamYO3eu1CYxMRENDQ1YuHAhqqqqEBERgZycHHh4eEht1q1bBwcHB8ycORMNDQ2IiYnB1q1bYW9vb4ndslqDBg0yqk4kV1lZmcEAAIC6ujqUlZVJZwzIOArBERbU1NRApVJBp9NxfKATf//737F+/foO64sWLcLDDz9sxh6RtRJC4IUXXjB4hdCIESPw9ttv85RQF+S+r1l82gjqO65du2ZUncgU+LnVtBgCJNv+/fuNqhPJVVpa2ul9Am1XEJLxGAIkW1eDcRysI1MJCAjAmDFjYGen/xZlb2+P8PBw/q6ZEEOAZHNw6Pw6gq7qRHIpFAosXry43akfIQQWL17M8QATYgiQbNOmTTOqTtQdfn5+8Pb21tvm7e2NwYMHW6hH1okhQLJ99dVXRtWJuuPw4cO4fPmy3rZLly7h8OHDFuqRdWIIkGxdLdTDhXzIVFpbW5GUlGSwlpSUhNbWVvN2yIoxBEi2ruZw5xzvZCoFBQWd3ixWUFBg5h5ZL4YAyRYTE2NUnUgu3p1uPgwBkq2j9Rnk1onkGjJkiN4qg78UFBTUYY26jyFAsl26dMmoOlF3ODk5Gdzu6Oho5p5YN4YAyRYaGmpUnUiukpISnD171mDt7NmzXFnMhBgCJFtHt/HLrRNR78MQINl8fHyMqhPJFRgYiDvuuMNgbdiwYZxG2oQYAiTbzfO4dLdOZAqcRdS0+FdLsmk0GqPqRHJxTMB8GAIkG9d9JXPp6tM+jwZMhyFARL0OP3CYD0OAZOOnMzKXgIAAODs7G6w5OztzPQETYgiQbBUVFUbVieQqKSlBY2OjwVpjYyPHBEyIIUCy+fr6GlUnkosfOMyHIUCynTx50qg6kVxjx46Fq6urwZqrqyvGjh1r5h5ZL4YAyTZ58mSj6kRyKRSKDqcm9/Pz48CwCTEESDYuKkPmUlpa2ul9AqWlpWbukfWyaAgkJSVBoVDoPdRqtVQXQiApKQkajQYuLi6Ijo7GqVOn9J6jqakJixYtgo+PD9zc3DB16lS+Gd0iXa3mxNWeyFQCAgIwZswYg7Xw8HBeHWRCFj8SGD58OCoqKqTHLychW716NdauXYu0tDQUFhZCrVZjwoQJqK2tldokJCQgOzsbWVlZOHDgAOrq6jB58mS0tLRYYnes2sWLF42qE8mlUCgwa9Ysg7VZs2bxdJAJWTwEHBwcoFarpceAAQMA3DgKSE1NxYoVKzBjxgyEhIQgIyMD9fX1yMzMBADodDqkp6djzZo1iI2NxahRo7B9+3acOHEC+/bts+RuWSXewEPmIoTAzp07DdaysrJ4T4oJWTwEfvjhB2g0GgQFBeGxxx7DuXPnAADFxcXQarWIi4uT2iqVSkRFRSE/Px8AUFRUhGvXrum10Wg0CAkJkdoY0tTUhJqaGr0HdS0iIsKoOpFcpaWlKCwsNFgrLCzkmIAJWTQEIiIi8P7772Pv3r3YtGkTtFotIiMj8fPPP0tLFd587bmvr69U02q1cHJyQv/+/TtsY0hKSgpUKpX08Pf3N/GeWaeO/ijl1onk8vf3h6enp8Gap6cn/2ZNyKIhMHHiRDz88MMIDQ1FbGwsvvjiCwBARkaG1ObmUwxCiC5PO3TVZvny5dDpdNKjrKzMiL2wHR0N1MmtE8lVVlbW4RF6TU0N/2ZNyOKng37Jzc0NoaGh+OGHH6SrhG7+RF9ZWSkdHajVajQ3N6OqqqrDNoYolUp4enrqPahrBw8eNKpOJFdAQECHy5WOGDGCVweZUK8KgaamJnz//fcYNGgQgoKCoFarkZubK9Wbm5uRl5eHyMhIAEBYWBgcHR312lRUVODkyZNSGzKdrq644hVZZA4cFDYtB0u++LJlyzBlyhQEBASgsrISf/7zn1FTU4P58+dDoVAgISEBycnJCA4ORnBwMJKTk+Hq6oo5c+YAAFQqFeLj47F06VJ4e3vDy8sLy5Ytk04vkWldunTJqDqRXKWlpR2uWX3ixAmUlpZyiUkTsWgIlJeXY/bs2bh8+TIGDBiAsWPHoqCgQPrPTUxMRENDAxYuXIiqqipEREQgJycHHh4e0nOsW7cODg4OmDlzJhoaGhATE4OtW7fC3t7eUrtltXizGJlL28CwoXEBDgyblkVDICsrq9O6QqFAUlISkpKSOmzj7OyM9evXY/369SbuHd2sq2Bl8JKpyBkY5pGAafSqMQHq3aZMmWJUnUiugIAA3HHHHQZrw4YN48CwCTEESLZDhw4ZVSeSSwjR4Q1hJSUlHBw2IYYAyXb9+nWj6kRyffvtt52uLPbtt9+auUfWiyFAsnU0ta/cOpFcvBzZfBgCJFtwcLBRdSK5Ll++bFSd5GMIkGw335nd3TqRXCNHjjSqTvIxBEg2Li9J5sLLkc2HIUCyHT582Kg6kVwBAQFwd3c3WHN3d+cloibEECDZwsPDO/wEZm9vj/DwcDP3iKxVWVkZ6urqDNbq6uo4i6gJMQRItrKysg6vymhpaeEfJpkM1xMwH4YAydbVDTq8gYdMhesJmA9DgGTjGsNkLn5+fp2eevTz8zNzj6wXQ4BkCwwM7HQ+F07oRaZy6NChTk89cooS02EIULcolUqD252cnMzcE7JmXMrUfHocAtu2bcN9990HjUaDkpISAEBqaio++eQTk3WOehc5C30QmQKXMjWfHoXAhg0bsGTJEjz00EOorq6WDtv69euH1NRUU/aPehE/Pz/Y2Rn+lbGzs+N5WjIZXoRgPj0KgfXr12PTpk1YsWKF3uDN6NGjO/ykSH1fQUFBh6uHtba2oqCgwMw9Imul0WiMqpN8PQqB4uJijBo1qt12pVKJq1evGt0p6p24vCSZS0dHnHLrJF+PfpJBQUE4duxYu+27d+/G3XffbWyfqJfiHyaZS0BAAFxdXQ3WXF1dOW2ECfVojeE//vGPeO6559DY2AghBA4dOoQdO3YgJSUFmzdvNnUfqZcYN24cnJ2dDS724ezsjHHjxlmgV2SNSktLUV9fb7BWX1+P0tJSDBkyxLydslI9CoHf/e53uH79OhITE1FfX485c+Zg8ODBePvtt/HYY4+Zuo/USygUCgQEBBhcPCYwMJA3ixH1QT0+fv/973+PkpISVFZWQqvVoqysDPHx8absG/UypaWlHa4edubMGV4iSiYTGBiI0NBQg7URI0bwxkQT6lEINDQ0SIdqPj4+aGhoQGpqKnJyckzaOepdOKkXmYtCocBLL73U7uiyo+3Ucz0KgWnTpuH9998HAFRXVyM8PBxr1qzBtGnTsGHDBpN2kHoPTupF5uTn59fuQpPhw4dj8ODBFuqRdepRCBw5cgT3338/AOCjjz6CWq1GSUkJ3n//fbzzzjs96khKSgoUCgUSEhKkbUIIJCUlQaPRwMXFBdHR0Th16pTe9zU1NWHRokXw8fGBm5sbpk6divLy8h71gToXEBCAMWPGGPx0Fh4ezis2yKTKy8vx/fff6237/vvv+fdtYj0Kgfr6enh4eAAAcnJyMGPGDNjZ2WHs2LHSFBLdUVhYiI0bN2LEiBF621evXo21a9ciLS0NhYWFUKvVmDBhAmpra6U2CQkJyM7ORlZWFg4cOIC6ujpMnjy5w8mnqOcUCgUWL17c4XYeopOpCCHw9ttvd7iddwybTo9CYOjQodi1axfKysqwd+9exMXFAQAqKys7PGfckbq6OsydOxebNm1C//79pe1CCKSmpmLFihWYMWMGQkJCkJGRgfr6emRmZgIAdDod0tPTsWbNGsTGxmLUqFHYvn07Tpw4gX379vVk16gLfn5+CAkJ0dsWEhLCQ3QyqdLSUhQWFra7AbG1tRWFhYW8CMGEehQCr776KpYtW4YhQ4YgPDxcuj48JyfH4J3EnXnuuecwadIkxMbG6m0vLi6GVquVAga4cUdyVFQU8vPzAQBFRUW4du2aXhuNRoOQkBCpjSFNTU2oqanRe5A85eXlOH36tN6206dP8xCdTCogIKDTact56tF0ehQCjzzyCEpLS3H48GHs3btX2h4TE4N169bJfp6srCwcOXIEKSkp7WparRYA4Ovrq7fd19dXqmm1Wjg5OekdQdzcxpCUlBSoVCrpwata5OEhOpmLEAIXL140WLtw4QJ/10yox/cJqNVqjBo1ChcvXsSFCxcA3FiI/M4775T1/WVlZVi8eDG2b98OZ2fnDtvdfJ5ZCNHlueeu2ixfvhw6nU568KoWedoO0W8eb2lpaeEhOpnUwYMHO11onlNJm06PQqC1tRUrV66ESqVCYGAgAgIC0K9fP/zv//6v7EnEioqKUFlZibCwMDg4OMDBwQF5eXl455134ODgIB0B3PyJvrKyUqqp1Wo0NzejqqqqwzaGKJVKeHp66j2oa21XB9287J+9vT2vDiKTCg8P73R5yfDwcDP3yHr1KARWrFiBtLQ0vPnmmzh69CiOHDmC5ORkrF+/Hq+88oqs54iJicGJEydw7Ngx6TF69GjMnTsXx44dw2233Qa1Wo3c3Fzpe5qbm5GXl4fIyEgAQFhYGBwdHfXaVFRU4OTJk1IbMh1eHUTmUl5e3unykhyDMp0ezR2UkZGBzZs3Y+rUqdK2kSNHYvDgwVi4cCHeeOONLp/Dw8Oj3VUmbm5u8Pb2lrYnJCQgOTkZwcHBCA4ORnJyMlxdXTFnzhwAgEqlQnx8PJYuXQpvb294eXlh2bJlCA0NbTfQTKbh5+eHOXPmYPv27dJpt7a5o4hMpe2os7CwsF2NR52m1aMjgStXrhg893/nnXfiypUrRneqTWJiIhISErBw4UKMHj0aFy5cQE5OjnSPAgCsW7cO06dPx8yZM3HffffB1dUVn332WYeHkmS8uXPnwtvbG8CNaUPaQpnIVDo66gTAo04TU4geDLNHREQgIiKi3d3BixYtQmFhYZ9bYaqmpgYqlQo6nY7jAzLl5+fj7bffxuLFi3nqjW6J8vJyzJs3T+9KIIVCgW3btnEpUxnkvq/1KATy8vIwadIkBAQEYNy4cVAoFMjPz0dZWRn+8Y9/SFNK9BUMAaLeRQiBxMREFBUV6V1sYmdnh7CwMKxevZpHA12Q+77Wo9NBUVFROHv2LH7729+iuroaV65cwYwZM3DmzJk+FwBE1PvwjmHz6dHAMHDjzlw5A8BERN3VNjB85MgRvauE7O3tERYWxoFhE5IdAsePH5f9pDdPBEdE1B1tA8Pz5883uJ2ngkxHdgjcc889UCgUXd6urVAoOIMnERmNlyObh+wQKC4uvpX9ICJqZ+7cudi9ezcuX77My5FvEdkDw4GBgdIjMzMTX375pd62wMBAfPnll8jKyrqV/SUiG+Ls7IwlS5bA19cXL774YqfzjFHP9OgS0SFDhiAzM7Pd9eEHDx7EY4891ueOGniJKBFZm1t6iahWq8WgQYPabR8wYAAqKip68pRERGQBPQoBf39//Otf/2q3/V//+hc0Go3RnSIiIvPo0X0CTz31FBISEnDt2jWMHz8eAPDll18iMTERS5cuNWkHiYjo1ulRCCQmJuLKlStYuHAhmpubAdwYwHnppZewfPlyk3aQiIhunR4NDLepq6vD999/DxcXFwQHB0OpVJqyb2bDgWEisjZy39d6PG0EALi7u2PMmDHGPAUREVlQj9cYJiKivo8hQERkwxgCREQ2jCFARGTDGAJERDaMIUBEZMMYAkRENowhQERkwxgCREQ2jCFARGTDGAJERDbMoiGwYcMGjBgxAp6envD09MS4ceOwe/duqS6EQFJSEjQaDVxcXBAdHY1Tp07pPUdTUxMWLVoEHx8fuLm5YerUqSgvLzf3rhAR9UkWDQE/Pz+8+eabOHz4MA4fPozx48dj2rRp0hv96tWrsXbtWqSlpaGwsBBqtRoTJkxAbW2t9BwJCQnIzs5GVlYWDhw4gLq6OkyePBktLS2W2i0ior5D9DL9+/cXmzdvFq2trUKtVos333xTqjU2NgqVSiXee+89IYQQ1dXVwtHRUWRlZUltLly4IOzs7MSePXs6fI3Gxkah0+mkR1lZmQAgdDrdrdsxIiIz0ul0st7Xes2YQEtLC7KysnD16lWMGzcOxcXF0Gq1iIuLk9oolUpERUUhPz8fAFBUVIRr167ptdFoNAgJCZHaGJKSkgKVSiU9/P39b92OERH1YhYPgRMnTsDd3R1KpRLPPPMMsrOzcffdd0Or1QIAfH199dr7+vpKNa1WCycnJ/Tv37/DNoYsX74cOp1OepSVlZl4r4iI+gajFpUxhWHDhuHYsWOorq7G3//+d8yfPx95eXlSXaFQ6LUXQrTbdrOu2iiVyj67ChoRkSlZ/EjAyckJQ4cOxejRo5GSkoKRI0fi7bffhlqtBoB2n+grKyulowO1Wo3m5mZUVVV12IaIiDpm8RC4mRACTU1NCAoKglqtRm5urlRrbm5GXl4eIiMjAQBhYWFwdHTUa1NRUYGTJ09KbYiIqGMWPR30pz/9CRMnToS/vz9qa2uRlZWF/fv3Y8+ePVAoFEhISEBycjKCg4MRHByM5ORkuLq6Ys6cOQAAlUqF+Ph4LF26FN7e3vDy8sKyZcsQGhqK2NhYS+4aEVGfYNEQ+OmnnzBv3jxUVFRApVJhxIgR2LNnDyZMmAAASExMRENDAxYuXIiqqipEREQgJycHHh4e0nOsW7cODg4OmDlzJhoaGhATE4OtW7fC3t7eUrtFRNRnKIQQwtKdsLSamhqoVCrodDp4enpaujtEREaT+77W68YEiIjIfBgCREQ2jCFARGTDGAJERDaMIUBEZMMYAkRENowhQERkwxgCREQ2jCFARGTDGAJERDaMIUBEZMMYAkRENowhQERkwxgCREQ2jCFARGTDGAJERDaMIUBEZMMYAkRENowhQERkwxgCREQ2jCFARGTDGAJERDaMIUBEZMMYAkRENsyiIZCSkoIxY8bAw8MDAwcOxPTp03HmzBm9NkIIJCUlQaPRwMXFBdHR0Th16pRem6amJixatAg+Pj5wc3PD1KlTUV5ebs5dISLqkywaAnl5eXjuuedQUFCA3NxcXL9+HXFxcbh69arUZvXq1Vi7di3S0tJQWFgItVqNCRMmoLa2VmqTkJCA7OxsZGVl4cCBA6irq8PkyZPR0tJiid0iIuo7RC9SWVkpAIi8vDwhhBCtra1CrVaLN998U2rT2NgoVCqVeO+994QQQlRXVwtHR0eRlZUltblw4YKws7MTe/bskfW6Op1OABA6nc6Ee0NEZDly39d61ZiATqcDAHh5eQEAiouLodVqERcXJ7VRKpWIiopCfn4+AKCoqAjXrl3Ta6PRaBASEiK1uVlTUxNqamr0HkREtqjXhIAQAkuWLMGvfvUrhISEAAC0Wi0AwNfXV6+tr6+vVNNqtXByckL//v07bHOzlJQUqFQq6eHv72/q3SEi6hN6TQg8//zzOH78OHbs2NGuplAo9L4WQrTbdrPO2ixfvhw6nU56lJWV9bzjRER9WK8IgUWLFuHTTz/FV199BT8/P2m7Wq0GgHaf6CsrK6WjA7VajebmZlRVVXXY5mZKpRKenp56DyIiW2TREBBC4Pnnn8fHH3+Mf/7znwgKCtKrBwUFQa1WIzc3V9rW3NyMvLw8REZGAgDCwsLg6Oio16aiogInT56U2hARkWEOlnzx5557DpmZmfjkk0/g4eEhfeJXqVRwcXGBQqFAQkICkpOTERwcjODgYCQnJ8PV1RVz5syR2sbHx2Pp0qXw9vaGl5cXli1bhtDQUMTGxlpy94iIej2LhsCGDRsAANHR0Xrbt2zZggULFgAAEhMT0dDQgIULF6KqqgoRERHIycmBh4eH1H7dunVwcHDAzJkz0dDQgJiYGGzduhX29vbm2hUioj5JIYQQlu6EpdXU1EClUkGn03F8gIisgtz3tV4xMExERJbBECAismEMASIiG8YQICKyYQwBIiIbxhAgIrJhDAEiIhvGECAismEMASIiG8YQICKyYQwB6pH8/HzMmjWrw9XbiKhvYAhQtzU2NiIlJQU//fQTUlJS0NjYaOkuEVEPMQSo2zIyMlBbWwsAqK2txfvvv2/hHhFRTzEEqFvKy8vbLQGamZmJ8vJyC/WIiIzBECDZhBBYtWqVwdqqVavAWcmJ+h6GAMlWUlKCEydOGKydOHECJSUlZu4RERmLIUCytba2GlUn6on09HSMHz8e6enplu6KVWIIkGxta0D3tE7UXdXV1di2bRtaW1uxbds2VFdXW7pLVochQLKFhYUZVSfqriVLluh9vXTpUgv1xHoxBEi2N954w6g6UXccPnwY586d09v2n//8B4cPH7ZQj6wTQ4Bk+/rrr42qE8nV2tqK5cuXG6wtX76c408mxBAgol7nm2++wbVr1wzWrl27hm+++cbMPbJeDAEi6nW+/PJLo+okH0OAZPPw8DCqTiRXVFSUUXWSz6Ih8PXXX2PKlCnQaDRQKBTYtWuXXl0IgaSkJGg0Gri4uCA6OhqnTp3Sa9PU1IRFixbBx8cHbm5umDp1KqcwuEXq6+uNqhPJdeTIEaPqJJ9FQ+Dq1asYOXIk0tLSDNZXr16NtWvXIi0tDYWFhVCr1ZgwYYI0eRkAJCQkIDs7G1lZWThw4ADq6uowefJktLS0mGs3bAZvFiNzCQwMNKpO8jlY8sUnTpyIiRMnGqwJIZCamooVK1ZgxowZAG7MXunr64vMzEw8/fTT0Ol0SE9Px7Zt2xAbGwsA2L59O/z9/bFv3z48+OCDZtsXW+Dq6oqrV692WicyBW9vb6PqJF+vHRMoLi6GVqtFXFyctE2pVCIqKkpayKSoqAjXrl3Ta6PRaBASEtLpYidNTU2oqanRe1DXZs+ebVSdSK6PPvrIqDrJ12tDoG0KAl9fX73tvr6+Uk2r1cLJyQn9+/fvsI0hKSkpUKlU0sPf39/EvbdOdXV1RtWJ5BoxYoRRdZKv14ZAG4VCofe1EKLdtpt11Wb58uXQ6XTSo6yszCR9tXY3D8p3t04kV3x8vFF1kq/XhoBarQbQflKyyspK6ehArVajubkZVVVVHbYxRKlUwtPTU+9BXeMEcmQuhw4dMqpO8vXaEAgKCoJarUZubq60rbm5GXl5eYiMjARwY8IyR0dHvTYVFRU4efKk1IZM59KlS0bVieSqqKgwqk7yWfTqoLq6Ovz444/S18XFxTh27Bi8vLwQEBCAhIQEJCcnIzg4GMHBwUhOToarqyvmzJkDAFCpVIiPj8fSpUvh7e0NLy8vLFu2DKGhodLVQmQ6np6enQ6i84iKTOWX7ws9qZN8Fg2Bw4cP44EHHpC+bps2dv78+di6dSsSExPR0NCAhQsXoqqqChEREcjJydG7M3XdunVwcHDAzJkz0dDQgJiYGGzduhX29vZm3x9rFxsbi48//rjTOpEpHDx40Kg6yWfREIiOju50XVqFQoGkpCQkJSV12MbZ2Rnr16/H+vXrb0EP6ZcGDx5sVJ1IrqioKHzyySed1sk0eu2YAPU+06dPN6pOJFdXd/xzRgDTYQiQbPb29nB0dDRYc3R05Ck4MpnFixcbVSf5GAIk25UrVzqd4/3KlStm7hFZK0dHR0yaNMlgbcqUKR1+GKHuYwiQbG1zOPW0TiSXEAKnT582WDt58mSnY4nUPQwBkk2j0RhVJ5KruLgYxcXF3a5R9zEESLbKykqj6kRyHT161Kg6yccQINnc3NyMqhPJ1dX8YF3VST6L3idAPSeEQGNjo1lfU6fTdVlvaGgwU29u3CPCNwPrNG3aNKxfv97guX87OztMmzbNAr2yTgyBPqqxsbHDBXksyZx92r17N1xcXMz2emQ+dnZ2UCqVBj/oODk5wc6OJzFMhT9JIup1fvjhhw6PdBsbG/HDDz+YuUfWSyF4rRVqamqgUqmg0+n6zCRoljgd1Nzc3Olh+CeffAInJyez9Yeng6zX008/jTNnznRYHzZsGP7617+asUd9j9z3NZ4O6qMUCoXZT4W4uLhg0qRJ+OKLL9rVpkyZApVKZdb+kHlY4gNHdXV1l3WOP5kGjwTQN48ELCk6Orrdtv3795u9H2QeDQ0NvXL8yZz64viT3Pc1jglQt61Zs0bv67S0NAv1hIiMxdNB1G1333239G8fHx+EhIRYsDd0qzk7O2P37t1mfc2SkhI888wzHdbfe+89BAYGmq0/zs7OZnstc2MIkFG2bdtm6S7QLWaJ8ac777zTqDrJx9NBRNQrdTTOxPEn0+KRgBEscdVEb/DLfbbF/Qes+2qR3iQkJAQnT56Uvh45cqQFe2OdeHUQen51EK+asF198WqRvujmvzEeBcjH+wSIrAiPOoHs7Gyz3hvQW9zqo06GgInU3TMbws5GfpxCAK3Xb/zbzgGwkdMiitbrcD+2wyKv3VvnijKn3/72t5bugkXc6qNOG3nXuvWEnQNgb0tL3plveojewubPm5JVYggYQW84pcXw2rtkRX7xf2zJobS0X12B0t42IkkIoLn1xr+d7GzmoBNNLQo8f8DLLK/FEDBCU1OT9G+P77Is2BMyt6amJri6uprt9fRDxzYCALjxpq+0t3QvLOG//8e3+gOH1YTAu+++i7feegsVFRUYPnw4UlNTcf/991u6W0Qm8csPHM8f8LZgT8jcbvUHDqu4WWznzp1ISEjAihUrcPToUdx///2YOHEiSktLb+nrKpXKW/r81Hvx/56shVWEwNq1axEfH4+nnnoKd911F1JTU+Hv748NGzbc0tflzUK2y9z/9wwd23Wr/+/7/Omg5uZmFBUV4eWXX9bbHhcXh/z8fIPf09TUpHd4XVNT06PXtsTEWm0aGxtt9pK5NtnZ2Rab2Mvcr8sPHLbrVv/f9/kQuHz5MlpaWuDr66u33dfXF1qt1uD3pKSk4PXXXzf6tS0xsRb9l7Ozs838/PmBw7Ks+QNHnw+BNjenpRCiwwRdvnw5lixZIn1dU1MDf3//W9o/U7Pkm4IQQjqSUiqVFvuUas3T+97Mkh84+Ltm3XNF9fkQ8PHxgb29fbtP/ZWVle2ODtoolco+f47V0kch5rw8kiyLv2vWrc8PDDs5OSEsLAy5ubl623NzcxEZGWmhXhER9Q19/kgAAJYsWYJ58+Zh9OjRGDduHDZu3IjS0tJOVyYiIiIrCYFZs2bh559/xsqVK1FRUYGQkBD84x//MOvyc0REfRHXE0DP1xMgIuqt5L6v9fkxASIi6jmGABGRDWMIEBHZMIYAEZENYwgQEdkwhgARkQ2zivsEjNV2lWxPZxMlIupt2t7PuroLgCEAoLa2FgD63CRyRERdqa2thUql6rDOm8UAtLa24uLFi/Dw8LDamQJNrW3m1bKyMt5gR7cUf9d6RgiB2tpaaDQa2Nl1fOafRwIA7Ozs4OfnZ+lu9Emenp78wySz4O9a93V2BNCGA8NERDaMIUBEZMMYAtQjSqUSr732Wp9fnId6P/6u3VocGCYismE8EiAismEMASIiG8YQICKyYQwBIiIbxhCgHnn33XcRFBQEZ2dnhIWF4ZtvvrF0l8gKff3115gyZQo0Gg0UCgV27dpl6S5ZHYYAddvOnTuRkJCAFStW4OjRo7j//vsxceJElJaWWrprZGWuXr2KkSNHIi0tzdJdsVq8RJS6LSIiAvfeey82bNggbbvrrrswffp0pKSkWLBnZM0UCgWys7Mxffp0S3fFqvBIgLqlubkZRUVFiIuL09seFxeH/Px8C/WKiHqKIUDdcvnyZbS0tMDX11dvu6+vL7RarYV6RUQ9xRCgHrl5ym0hBKfhJuqDGALULT4+PrC3t2/3qb+ysrLd0QER9X4MAeoWJycnhIWFITc3V297bm4uIiMjLdQrIuopLipD3bZkyRLMmzcPo0ePxrhx47Bx40aUlpbimWeesXTXyMrU1dXhxx9/lL4uLi7GsWPH4OXlhYCAAAv2zHrwElHqkXfffRerV69GRUUFQkJCsG7dOvz617+2dLfIyuzfvx8PPPBAu+3z58/H1q1bzd8hK8QQICKyYRwTICKyYQwBIiIbxhAgIrJhDAEiIhvGECAismEMASIiG8YQICKyYQwBIiIbxhAgIrJhDAGyeQqFotPHggULpLZxcXGwt7dHQUFBu+dZsGCB9D0ODg4ICAjAs88+i6qqqnZtjx49ilmzZmHQoEFQKpUIDAzE5MmT8dlnn6HtJv7z58932KeCggJER0d32u8hQ4bcqh8ZWRFOIEc2r6KiQvr3zp078eqrr+LMmTPSNhcXFwBAaWkpvv32Wzz//PNIT0/H2LFj2z3Xb37zG2zZsgXXr1/H6dOn8eSTT6K6uho7duyQ2nzyySeYOXMmYmNjkZGRgdtvvx0///wzjh8/jv/5n//B/fffj379+knt9+3bh+HDh+u9jre3Nz7++GM0NzcDAMrKyhAeHq7X1t7e3vgfDlk9hgDZPLVaLf1bpVJBoVDobWuzZcsWTJ48Gc8++yzCw8ORmpoKNzc3vTZKpVL6Xj8/P8yaNUtvorOrV68iPj4ekyZNwscffyxtv/322xEeHo6nnnoKN0/n5e3tbbA/Xl5e0r8bGxs7bUvUEZ4OIpJBCIEtW7bg8ccfx5133ok77rgDf/vb3zr9nnPnzmHPnj1wdHSUtuXk5ODnn39GYmJih9/HFdrInBgCRDLs27cP9fX1ePDBBwEAjz/+ONLT09u1+/zzz+Hu7g4XFxfcfvvtOH36NF566SWpfvbsWQDAsGHDpG2FhYVwd3eXHp9//rnec0ZGRurV3d3d0dLScit2k2wQTwcRyZCeno5Zs2bBweHGn8zs2bPxxz/+EWfOnNF7Q3/ggQewYcMG1NfXY/PmzTh79iwWLVrU6XOPGDECx44dAwAEBwfj+vXrevWdO3firrvu0tvG8/1kKjwSIOrClStXsGvXLrz77rtwcHCAg4MDBg8ejOvXr+P//b//p9fWzc0NQ4cOxYgRI/DOO++gqakJr7/+ulQPDg4GAL2BZ6VSiaFDh2Lo0KEGX9/f31+qd9aOqCcYAkRd+OCDD+Dn54fvvvsOx44dkx6pqanIyMho98n9l1577TX85S9/wcWLFwHcuMTUy8sLq1atMlf3iTrFECDqQnp6Oh555BGEhIToPdou//ziiy86/N7o6GgMHz4cycnJAAB3d3ds3rwZX3zxBSZNmoS9e/fi3LlzOH78OFavXg2g/amen3/+GVqtVu/RdjUQkbEYAkSdKCoqwnfffYeHH364Xc3DwwNxcXEGB4h/acmSJdi0aRPKysoAAL/97W+Rn58PV1dXPPHEExg2bBjGjx+Pf/7zn8jKysLkyZP1vj82NhaDBg3Se+zatctk+0i2jWsMExHZMB4JEBHZMIYAEZENYwgQEdkwhgARkQ1jCBAR2TCGABGRDWMIEBHZMIYAEZENYwgQEdkwhgARkQ1jCBAR2bD/D6ykI3Eqz5FWAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "numerical_features = ['high','close','low','open']\n",
    "for feature in numerical_features:\n",
    "    plt.figure(figsize=(4, 4))\n",
    "    sns.boxplot(x='TARGET', y=feature, data=df)\n",
    "    plt.title(f'{feature} vs Output')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04f9dca4-d24a-47e3-899b-ba1adab72c34",
   "metadata": {},
   "source": [
    "# Split the data into training and testing sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "9d472067-ded5-45b9-8dc2-12fc52bb0f86",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Found input variables with inconsistent numbers of samples: [4, 7781]",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[130], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m X_train, X_test, y_train, y_test\u001b[38;5;241m=\u001b[39mtrain_test_split(feature,target,test_size\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.4\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:211\u001b[0m, in \u001b[0;36mvalidate_params.<locals>.decorator.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    205\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m    206\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m    207\u001b[0m         skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m    208\u001b[0m             prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m    209\u001b[0m         )\n\u001b[0;32m    210\u001b[0m     ):\n\u001b[1;32m--> 211\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    212\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m InvalidParameterError \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m    213\u001b[0m     \u001b[38;5;66;03m# When the function is just a wrapper around an estimator, we allow\u001b[39;00m\n\u001b[0;32m    214\u001b[0m     \u001b[38;5;66;03m# the function to delegate validation to the estimator, but we replace\u001b[39;00m\n\u001b[0;32m    215\u001b[0m     \u001b[38;5;66;03m# the name of the estimator by the name of the function in the error\u001b[39;00m\n\u001b[0;32m    216\u001b[0m     \u001b[38;5;66;03m# message to avoid confusion.\u001b[39;00m\n\u001b[0;32m    217\u001b[0m     msg \u001b[38;5;241m=\u001b[39m re\u001b[38;5;241m.\u001b[39msub(\n\u001b[0;32m    218\u001b[0m         \u001b[38;5;124mr\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mparameter of \u001b[39m\u001b[38;5;124m\\\u001b[39m\u001b[38;5;124mw+ must be\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    219\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mparameter of \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfunc\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__qualname__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m must be\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    220\u001b[0m         \u001b[38;5;28mstr\u001b[39m(e),\n\u001b[0;32m    221\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:2614\u001b[0m, in \u001b[0;36mtrain_test_split\u001b[1;34m(test_size, train_size, random_state, shuffle, stratify, *arrays)\u001b[0m\n\u001b[0;32m   2611\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m n_arrays \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m   2612\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mAt least one array required as input\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m-> 2614\u001b[0m arrays \u001b[38;5;241m=\u001b[39m indexable(\u001b[38;5;241m*\u001b[39marrays)\n\u001b[0;32m   2616\u001b[0m n_samples \u001b[38;5;241m=\u001b[39m _num_samples(arrays[\u001b[38;5;241m0\u001b[39m])\n\u001b[0;32m   2617\u001b[0m n_train, n_test \u001b[38;5;241m=\u001b[39m _validate_shuffle_split(\n\u001b[0;32m   2618\u001b[0m     n_samples, test_size, train_size, default_test_size\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.25\u001b[39m\n\u001b[0;32m   2619\u001b[0m )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:455\u001b[0m, in \u001b[0;36mindexable\u001b[1;34m(*iterables)\u001b[0m\n\u001b[0;32m    436\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Make arrays indexable for cross-validation.\u001b[39;00m\n\u001b[0;32m    437\u001b[0m \n\u001b[0;32m    438\u001b[0m \u001b[38;5;124;03mChecks consistent length, passes through None, and ensures that everything\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    451\u001b[0m \u001b[38;5;124;03m    sparse matrix, or dataframe) or `None`.\u001b[39;00m\n\u001b[0;32m    452\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    454\u001b[0m result \u001b[38;5;241m=\u001b[39m [_make_indexable(X) \u001b[38;5;28;01mfor\u001b[39;00m X \u001b[38;5;129;01min\u001b[39;00m iterables]\n\u001b[1;32m--> 455\u001b[0m check_consistent_length(\u001b[38;5;241m*\u001b[39mresult)\n\u001b[0;32m    456\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m result\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:409\u001b[0m, in \u001b[0;36mcheck_consistent_length\u001b[1;34m(*arrays)\u001b[0m\n\u001b[0;32m    407\u001b[0m uniques \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39munique(lengths)\n\u001b[0;32m    408\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(uniques) \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[1;32m--> 409\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    410\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFound input variables with inconsistent numbers of samples: \u001b[39m\u001b[38;5;132;01m%r\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    411\u001b[0m         \u001b[38;5;241m%\u001b[39m [\u001b[38;5;28mint\u001b[39m(l) \u001b[38;5;28;01mfor\u001b[39;00m l \u001b[38;5;129;01min\u001b[39;00m lengths]\n\u001b[0;32m    412\u001b[0m     )\n",
      "\u001b[1;31mValueError\u001b[0m: Found input variables with inconsistent numbers of samples: [4, 7781]"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test=train_test_split(feature,target,test_size=0.4)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7435fa6d-cdde-49ac-9a0c-1ff0ac455c03",
   "metadata": {},
   "source": [
    "# Create and train the RandomForestClassifier model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61c0c543-19b5-4e88-849f-b5236caefe87",
   "metadata": {},
   "source": [
    "# Reason for using this model :\n",
    "\n",
    "- >Robustness to Overfitting : Random Forest reduces the risk of overfitting by aggregating the predictions of multiple decision trees.\n",
    "- >Robust to Noise :  Random Forests are less sensitive to noisy data.\n",
    "- >Parallelization : The training of individual trees in a Random Forest can be efficient for large datasets typical in stock market analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "edaf6e3b-c3d5-4e04-94bf-211a2e9a3993",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7877152402981239"
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model=RandomForestClassifier(n_estimators=100,criterion='gini',)\n",
    "model.fit(X_train,y_train)\n",
    "model.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "id": "85f84371-dd96-45d6-9412-159cae36568a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\princ\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:464: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0], dtype=int64)"
      ]
     },
     "execution_count": 140,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predicts=[2.3,2.8,5.6,5.5]\n",
    "inputs=np.asarray(predicts).reshape(1,-1)\n",
    "model.predict(inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e30aa8df-c7c7-471c-8471-f8756ebcacbb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
