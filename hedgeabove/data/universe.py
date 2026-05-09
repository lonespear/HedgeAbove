"""Stock ticker universe — S&P 500, Russell 2000, and international ADRs."""


def _noop_cache(*args, **kwargs):
    def deco(fn):
        return fn
    return deco


def _resolve_cache_decorator():
    """Match data/market.py: real Streamlit cache when running under
    `streamlit run`, no-op decorator otherwise so the headless scorer /
    cron / Pi deployments don't pull in Streamlit."""
    try:
        import streamlit as st
        from streamlit.runtime import exists as _runtime_exists
        if _runtime_exists():
            return st.cache_data
    except Exception:
        pass
    return _noop_cache


_cache_data = _resolve_cache_decorator()


@_cache_data(ttl=3600)
def get_sp500_tickers():
    """Get comprehensive S&P 500 constituents with full sector coverage (~500+ tickers)"""
    return [
        # Technology - Mega Cap (40 stocks)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
        'ADBE', 'NFLX', 'CRM', 'CSCO', 'INTC', 'AMD', 'TXN', 'QCOM', 'IBM', 'NOW',
        'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
        'PANW', 'ADSK', 'ANSS', 'WDAY', 'TEAM', 'DDOG', 'CRWD', 'ZS', 'SNOW', 'NET',

        # Technology - Software & Cloud (30 stocks)
        'MSFT', 'ORCL', 'SAP', 'SHOP', 'SQ', 'TWLO', 'DOCU', 'ZM', 'OKTA', 'SPLK',
        'VEEV', 'RNG', 'HUBS', 'ZI', 'BILL', 'MNDY', 'PATH', 'GTLB', 'S', 'ESTC',
        'MDB', 'CFLT', 'DT', 'DOCN', 'FROG', 'PD', 'NCNO', 'ASAN', 'PCOR', 'BRZE',

        # Technology - Semiconductors (25 stocks)
        'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'ADI', 'MCHP', 'KLAC', 'LRCX',
        'AMAT', 'MU', 'NXPI', 'MRVL', 'ON', 'MPWR', 'SWKS', 'QRVO', 'ENTG', 'ALGM',
        'WOLF', 'SLAB', 'CRUS', 'SITM', 'LSCC',

        # Healthcare - Pharma & Biotech (50 stocks)
        'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'REGN', 'VRTX', 'HUM', 'ISRG', 'ZTS',
        'BIIB', 'MRNA', 'ILMN', 'IQV', 'BSX', 'MDT', 'SYK', 'EW', 'IDXX', 'HCA',
        'A', 'ALGN', 'ALNY', 'BAX', 'BDX', 'BIO', 'CNC', 'CTLT', 'DGX', 'DVA',
        'EXAS', 'GEHC', 'HOLX', 'HSIC', 'INCY', 'LH', 'MCK', 'MOH', 'PODD', 'RMD',

        # Financials - Banks (40 stocks)
        'JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SCHW', 'AXP', 'SPGI',
        'CB', 'MMC', 'PGR', 'TFC', 'USB', 'PNC', 'COF', 'BK', 'AIG', 'MET',
        'CME', 'ICE', 'MCO', 'AON', 'TRV', 'ALL', 'AFL', 'PRU', 'HIG', 'CINF',
        'BRO', 'L', 'GL', 'WRB', 'RJF', 'NTRS', 'CFG', 'HBAN', 'RF', 'KEY',

        # Financials - Insurance & Asset Management (20 stocks)
        'BRK.B', 'BLK', 'TROW', 'BEN', 'IVZ', 'STT', 'AMG', 'SEIC', 'EVR', 'PFG',
        'FNF', 'FAF', 'JKHY', 'CBOE', 'NDAQ', 'MKTX', 'MSCI', 'FDS', 'TW', 'VIRT',

        # Consumer Discretionary - Retail (40 stocks)
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG',
        'MAR', 'GM', 'F', 'ABNB', 'CMG', 'YUM', 'DRI', 'ROST', 'DG', 'ULTA',
        'EBAY', 'ETSY', 'W', 'BBY', 'DKS', 'FIVE', 'OLLI', 'BJ', 'BBWI', 'AEO',
        'ANF', 'BOOT', 'BWA', 'CRI', 'DDS', 'FL', 'GES', 'GPS', 'KSS', 'M',

        # Consumer Discretionary - Auto & Leisure (25 stocks)
        'TSLA', 'GM', 'F', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'HMC', 'TM',
        'RACE', 'STLA', 'CCL', 'RCL', 'NCLH', 'LVS', 'WYNN', 'MGM', 'CZR', 'PENN',
        'DKNG', 'FLUT', 'BALY', 'RSI', 'LYV',

        # Consumer Staples (35 stocks)
        'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'CL', 'MDLZ', 'KMB',
        'GIS', 'KHC', 'TSN', 'HSY', 'K', 'CLX', 'SJM', 'CPB', 'CAG', 'HRL',
        'MKC', 'CHD', 'TAP', 'STZ', 'BF.B', 'SAM', 'KDP', 'MNST', 'CELH', 'KR',
        'SYY', 'COKE', 'FLO', 'INGR', 'POST',

        # Energy (35 stocks)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        'WMB', 'KMI', 'BKR', 'HES', 'DVN', 'FANG', 'MRO', 'APA', 'CTRA', 'OVV',
        'NOV', 'FTI', 'CHX', 'RIG', 'VAL', 'PR', 'EQT', 'AR', 'MTDR', 'SM',
        'MGY', 'CNX', 'RRC', 'CIVI', 'CLB',

        # Industrials (50 stocks)
        'UNP', 'HON', 'RTX', 'UPS', 'CAT', 'DE', 'BA', 'LMT', 'GE', 'MMM',
        'FDX', 'NSC', 'EMR', 'ETN', 'ITW', 'PH', 'WM', 'CSX', 'NOC', 'GD',
        'PCAR', 'JCI', 'CARR', 'OTIS', 'TT', 'IR', 'FAST', 'ODFL', 'CHRW', 'JBHT',
        'EXPD', 'XPO', 'HUBG', 'GWW', 'WCC', 'DY', 'ALLE', 'BLDR', 'FBIN', 'VMI',
        'MLM', 'GNRC', 'AIT', 'AAON', 'ACM', 'ACA', 'AGCO', 'ALK', 'ARCB', 'B',

        # Communication Services (30 stocks)
        'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'EA',
        'TTWO', 'RBLX', 'U', 'PINS', 'SNAP', 'SPOT', 'MTCH', 'BMBL', 'YELP', 'ZG',
        'ROKU', 'PARA', 'WBD', 'FOXA', 'FOX', 'NWSA', 'NWS', 'NYT', 'OMC', 'IPG',

        # Utilities (25 stocks)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ES', 'ED',
        'PEG', 'EIX', 'WEC', 'AWK', 'DTE', 'PPL', 'FE', 'CMS', 'CNP', 'ATO',
        'NI', 'LNT', 'EVRG', 'PNW', 'OGE',

        # Real Estate - REITs (35 stocks)
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VTR', 'SBAC', 'INVH', 'ARE', 'MAA', 'DOC', 'UDR', 'ESS', 'KIM',
        'REG', 'FRT', 'BXP', 'VNO', 'SLG', 'HST', 'RHP', 'CPT', 'ELS', 'AMH',
        'CUBE', 'REXR', 'FR', 'STAG', 'TRNO',

        # Materials (30 stocks)
        'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC',
        'MLM', 'ALB', 'CTVA', 'EMN', 'MOS', 'CE', 'FMC', 'IFF', 'PKG', 'AMCR',
        'IP', 'SEE', 'AVY', 'BALL', 'CCK', 'NEM', 'GOLD', 'WPM', 'FNV', 'SCCO',

        # Payment/Fintech (20 stocks)
        'V', 'MA', 'PYPL', 'ADP', 'FIS', 'FISV', 'GPN', 'SQ', 'COIN', 'SOFI',
        'AFRM', 'UPST', 'LC', 'NU', 'HOOD', 'MELI', 'PAGS', 'STNE', 'PAYX', 'FLYW',

        # Aerospace & Defense (15 stocks)
        'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'HWM', 'TXT', 'HII',
        'AVAV', 'KTOS', 'AJRD', 'CW', 'SPR',

        # Emerging Growth & Innovation (20 stocks)
        'PLTR', 'IONQ', 'RKLB', 'SPCE', 'OPEN', 'DASH', 'UBER', 'LYFT', 'CVNA', 'CHWY',
        'CHEWY', 'W', 'FVRR', 'UPWK', 'ZI', 'DOCN', 'APPS', 'BIGC', 'SHOP', 'MELI',

        # Small-Cap Technology (100 stocks)
        'SMCI', 'DELL', 'HPQ', 'HPE', 'NTAP', 'WDC', 'STX', 'PSTG', 'CVLT', 'DBX',
        'BOX', 'MIME', 'TENB', 'VRNS', 'QLYS', 'RPD', 'PLAN', 'BLKB', 'BRZE', 'RELY',
        'ASAN', 'TASK', 'WDAY', 'PAYC', 'PCTY', 'PYCR', 'EEFT', 'EVTC', 'GDDY', 'WIX',
        'JFROG', 'AZPN', 'QTWO', 'SMAR', 'ENV', 'APPN', 'CWAN', 'NEWR', 'SUMO', 'SAIC',
        'LDOS', 'CACI', 'BAH', 'KBR', 'VRSK', 'TRU', 'GMED', 'TDC', 'FTV', 'ZBRA',
        'SWI', 'NATI', 'ROP', 'KEYS', 'TER', 'COHR', 'II', 'NOVT', 'LITE', 'VIAV',
        'FORM', 'DIOD', 'MKSI', 'ONTO', 'UCTT', 'PLAB', 'AOSL', 'CRUS', 'CEVA', 'XLNX',
        'MXIM', 'RMBS', 'SYNA', 'CCMP', 'SMTC', 'MTSI', 'COHU', 'LSCC', 'AMBA', 'MRVL',
        'CRUS', 'ACLS', 'MLAB', 'PI', 'AMBA', 'MPWR', 'POWI', 'VICR', 'ENPH', 'SEDG',
        'GNRC', 'RUN', 'NOVA', 'BLDR', 'ATKR', 'APH', 'TEL', 'GLW', 'JBL', 'FN',

        # Small-Cap Healthcare (100 stocks)
        'TECH', 'PEN', 'WAT', 'MTD', 'DXCM', 'RVTY', 'IQV', 'CRL', 'MEDP', 'SOLV',
        'VTRS', 'CORT', 'CPRX', 'PRGO', 'TEVA', 'SUPN', 'TMDX', 'KRYS', 'NTRA', 'ADMA',
        'HALO', 'RARE', 'FOLD', 'ION', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VERV', 'PRME',
        'BLUE', 'SAGE', 'NBIX', 'SRPT', 'BMRN', 'SGEN', 'JAZZ', 'UTHR', 'HZNP', 'ALKS',
        'ACAD', 'PTCT', 'RGNX', 'TBPH', 'ARVN', 'ARWR', 'MDGL', 'ITCI', 'KRTX', 'SAVA',
        'AGIO', 'APLS', 'DNLI', 'FATE', 'NRIX', 'VRTX', 'MRTX', 'NVCR', 'GLPG', 'IONS',
        'EXEL', 'BPMC', 'BGNE', 'LEGN', 'YMAB', 'IMMU', 'ESPR', 'PBYI', 'INSM', 'CLDX',
        'GOSS', 'RXRX', 'SDGR', 'ALLO', 'BCYC', 'MNOV', 'CDMO', 'VCEL', 'VCYT', 'QGEN',
        'NVST', 'LMNX', 'MYGN', 'NTRA', 'GKOS', 'SLP', 'NEOG', 'GTHX', 'KRYS', 'AVNS',
        'OMER', 'PCRX', 'ANIP', 'LBPH', 'AMRX', 'COLL', 'ETNB', 'SGMO', 'EDIT', 'PACB',

        # Small-Cap Financials (100 stocks)
        'EWBC', 'PACW', 'WAL', 'SIVB', 'SBNY', 'FRC', 'CMA', 'ZION', 'SNV', 'ONB',
        'UMBF', 'OZK', 'UBSI', 'HWC', 'ASB', 'FHN', 'BKU', 'FIBK', 'WSFS', 'TCBI',
        'CADE', 'SFNC', 'CASH', 'ABCB', 'VLY', 'PB', 'BANR', 'CATY', 'CBU', 'FFIN',
        'TBBK', 'SRCE', 'BPOP', 'FBK', 'FULT', 'INDB', 'WAFD', 'PFS', 'UCBI', 'BANF',
        'LKFN', 'BHLB', 'SFBS', 'HTLF', 'HOMB', 'CVBF', 'CBSH', 'BOKF', 'WTFC', 'ONB',
        'ENVA', 'BRKL', 'Ryan', 'BGC', 'VIRT', 'LPLA', 'SF', 'PIPR', 'APAM', 'HLNE',
        'LAZ', 'PJT', 'EVR', 'RYAN', 'JEF', 'MORN', 'MC', 'OMF', 'VRNT', 'CACC',
        'COOP', 'WRLD', 'TRUP', 'LMND', 'ROOT', 'MTTR', 'KNSL', 'BHF', 'ORI', 'RNR',
        'AFG', 'WTM', 'SAFT', 'KMPR', 'PLMR', 'THG', 'UFCS', 'NAVG', 'JRVR', 'IGIC',
        'AGII', 'STC', 'EIG', 'AMSF', 'PRA', 'HRTG', 'UFCS', 'HALL', 'TRIN', 'ANAT',

        # Small-Cap Consumer (100 stocks)
        'WING', 'BLMN', 'TXRH', 'EAT', 'CAKE', 'PLAY', 'DENN', 'RUTH', 'FWRG', 'BWLD',
        'SHAK', 'NDLS', 'PZZA', 'DAVE', 'BJRI', 'DIN', 'CBRL', 'BLMN', 'PNRA', 'BWLD',
        'SON', 'SONC', 'JACK', 'HAYW', 'ARKO', 'CASY', 'LAD', 'ABG', 'SAH', 'AN',
        'PAG', 'GPI', 'CRMT', 'MNRO', 'AAP', 'AZO', 'ORLY', 'BBWI', 'URBN', 'AEO',
        'PSMT', 'ANF', 'TLYS', 'CHS', 'GCO', 'HIBB', 'BGFV', 'CTRN', 'SHOO', 'WWW',
        'BOOT', 'DECK', 'CROX', 'VFC', 'RL', 'PVH', 'HBI', 'GIL', 'SCVL', 'GIII',
        'MOV', 'EXPR', 'ZUMZ', 'TLRD', 'JWN', 'M', 'KSS', 'DDS', 'CRI', 'BURL',
        'FIVE', 'DG', 'DLTR', 'BIG', 'OLLI', 'PRTY', 'CONN', 'BBBY', 'TCS', 'BEDS',
        'ASO', 'BGFV', 'HIBB', 'DKS', 'FL', 'SCVL', 'PIR', 'SIG', 'WGO', 'THO',
        'CWH', 'LCI', 'LCII', 'PATK', 'BC', 'POWL', 'REVG', 'SHYF', 'HCSG', 'CASY',

        # Small-Cap Energy (75 stocks)
        'PBF', 'DK', 'CIVI', 'CRC', 'CPE', 'WTI', 'OAS', 'AROC', 'VTLE', 'GPOR',
        'REI', 'TALO', 'CDEV', 'CLR', 'MTDR', 'FANG', 'PR', 'MUR', 'NBR', 'HP',
        'NINE', 'SWN', 'RRC', 'CNX', 'AR', 'CTRA', 'MGY', 'REPX', 'CRGY', 'GPRK',
        'LPI', 'GPRE', 'REX', 'VVV', 'AMPY', 'GRNT', 'ESTE', 'PTEN', 'PUMP', 'LBRT',
        'NEX', 'WTTR', 'TDW', 'WFRD', 'SDRL', 'VAL', 'TRGP', 'ENLC', 'PAGP', 'USAC',
        'GEL', 'DKL', 'EPD', 'ET', 'MMP', 'PAA', 'WES', 'AM', 'HESM', 'CEQP',
        'ENLC', 'NGL', 'SUN', 'SHLX', 'MPLX', 'PSX', 'NS', 'CQP', 'BP', 'E',
        'TOT', 'SHEL', 'ENB', 'TRP', 'CNQ',

        # Small-Cap Industrials (100 stocks)
        'JBHT', 'LSTR', 'KNX', 'SAIA', 'ARCB', 'WERN', 'ODFL', 'XPO', 'CVLG', 'YRCW',
        'GNK', 'SBLK', 'INSW', 'EGLE', 'SHIP', 'CMRE', 'EDRY', 'GOGL', 'NMM', 'SB',
        'TGH', 'MATX', 'KEX', 'HUBG', 'SNDR', 'FWRD', 'ECHO', 'MRTN', 'ULH', 'HTLD',
        'SNCY', 'RXO', 'GXO', 'JBLU', 'AAL', 'UAL', 'DAL', 'LUV', 'ALK', 'SAVE',
        'HA', 'SKYW', 'MESA', 'ATSG', 'AAWW', 'CYRX', 'ARCB', 'SNDR', 'JOBY', 'ACHR',
        'BLDE', 'LILM', 'EH', 'EVEX', 'EVTL', 'GEV', 'LEV', 'REE', 'WKHS', 'RIDE',
        'FSR', 'GOEV', 'ARVL', 'MULN', 'ELMS', 'OWLT', 'HYZN', 'NKLA', 'HYLN', 'GP',
        'RDN', 'MTG', 'ESNT', 'NMIH', 'HCI', 'UVE', 'NODK', 'PRA', 'UFCS', 'KNSL',
        'AIT', 'DY', 'WSO', 'MSM', 'RBC', 'TILE', 'FLS', 'BMI', 'CR', 'PRIM',
        'ATKR', 'AAON', 'AOS', 'AWI', 'AZEK', 'BCC', 'BECN', 'BLD', 'BXC', 'CSWI',

        # Small-Cap Materials (75 stocks)
        'MP', 'LAC', 'ALB', 'SQM', 'LTHM', 'PLL', 'SGML', 'LITM', 'NOVRF', 'CMP',
        'SMG', 'TUP', 'CATO', 'GFF', 'KOP', 'HWKN', 'FUL', 'HXL', 'SLVM', 'KWR',
        'NGVT', 'TROX', 'IOSP', 'NEU', 'SXT', 'CSTM', 'OMI', 'GEF', 'SON', 'SLGN',
        'MERC', 'RPM', 'AXTA', 'HUN', 'OLN', 'TSE', 'KRA', 'VVV', 'CBT', 'CC',
        'WLK', 'IOSP', 'PCT', 'ESNT', 'NGVT', 'BCPC', 'FUL', 'GRA', 'CENX', 'KALU',
        'ATI', 'ZEUS', 'HCC', 'HAYN', 'SYNL', 'MTUS', 'CRS', 'TMST', 'WOR', 'MTRN',
        'CMC', 'CLF', 'STLD', 'RS', 'X', 'MT', 'TX', 'ASTL', 'HEES', 'NEWP',
        'USLM', 'TGLS', 'HL', 'AG', 'CDE', 'EGO', 'PAAS', 'GPL', 'SVM', 'NGD',
        'AUY', 'SSRM', 'KGC', 'IAG', 'BTG', 'VALE', 'RIO', 'BHP', 'SCCO', 'FCX',
        'TECK', 'HBM', 'CMCL', 'VEDL', 'GLNCY',

        # Small-Cap Real Estate (75 stocks)
        'VRE', 'STWD', 'BXMT', 'AGNC', 'NLY', 'TWO', 'MITT', 'ARR', 'CIM', 'MFA',
        'NYMT', 'DX', 'PMT', 'EARN', 'IVR', 'RC', 'GPMT', 'ARI', 'TRTX', 'ORC',
        'LADR', 'AAIC', 'GPMT', 'LOAN', 'RC', 'BRMK', 'RWT', 'CHMI', 'EFC', 'WMC',
        'NRZ', 'RITM', 'AAMC', 'ABR', 'AJX', 'ACRE', 'MITT', 'KREF', 'TPVG', 'OXSQ',
        'INN', 'PEB', 'RLJ', 'SHO', 'PK', 'AHT', 'APLE', 'CHH', 'XHR', 'DRH',
        'CLDT', 'FCPT', 'GTY', 'JBGS', 'KRC', 'CUZ', 'DEI', 'HIW', 'SLG', 'BXP',
        'PGRE', 'PDM', 'CLI', 'VNO', 'ESRT', 'NYC', 'SVC', 'ALEX', 'BDN', 'CIO',
        'GOOD', 'RMAX', 'OPI', 'SRC', 'AAT', 'ADC', 'AKR', 'BFS', 'BRX', 'CDR',
        'CTRE', 'ELME', 'EPRT', 'LAND', 'LXP', 'NNN', 'NTST', 'OUT', 'ROIC', 'RPT',
        'SITC', 'UE', 'UMH', 'VRE', 'WPC',

        # Small-Cap Utilities (50 stocks)
        'AVA', 'AGR', 'ALE', 'AQN', 'ARTNA', 'BKH', 'CPK', 'CWEN', 'CWEN.A', 'NWE',
        'NWN', 'MDU', 'MGE', 'MSEX', 'OTTR', 'PNM', 'POR', 'SJW', 'SR', 'SWX',
        'UTL', 'YORW', 'BIP', 'NEP', 'AY', 'AWR', 'CDZI', 'CWCO', 'ELPC', 'GNE',
        'NOVA', 'NWE', 'NWN', 'OTTR', 'PNM', 'POR', 'SJW', 'SR', 'SWX', 'UTL',
        'UGI', 'NFE', 'CWEN', 'TAC', 'DUK', 'FTS', 'BEPC', 'AEP', 'CEG', 'VST',

        # Small-Cap Communication & Media (50 stocks)
        'SSTK', 'TGNA', 'LEE', 'NXST', 'GTN', 'SCHL', 'MSGS', 'FUBO', 'SATS', 'EVER',
        'GOGO', 'IRDM', 'GILT', 'VSAT', 'IRDM', 'CMCSA', 'CHTR', 'CABO', 'LBRDA', 'LBRDK',
        'LILA', 'LILAK', 'SIRI', 'LSXMA', 'LSXMB', 'LSXMK', 'GSAT', 'ASTS', 'SPCE', 'RKLB',
        'MAXN', 'PUBM', 'MGNI', 'TTD', 'APPS', 'BIGC', 'CRTO', 'NCMI', 'IMAX', 'CNK',
        'RGC', 'MSGN', 'WMG', 'SPOT', 'BMBL', 'MTCH', 'IAC', 'ANGI', 'CARS', 'CVNA',

        # Micro-Cap & Emerging Stocks (200 stocks)
        'AEHR', 'CLOV', 'GTLB', 'IOT', 'LUNR', 'PL', 'PTON', 'BROS', 'GRND', 'GPRO',
        'BYND', 'OUST', 'LAZR', 'LIDR', 'INVZ', 'MVIS', 'VLDR', 'AEYE', 'OLED', 'KOPN',
        'VUZI', 'WIMI', 'HIMX', 'GRMN', 'WOLF', 'SEDG', 'ENPH', 'RUN', 'ARRY', 'CSIQ',
        'DQ', 'FSLR', 'JKS', 'MAXN', 'NOVA', 'SOL', 'SPWR', 'SHLS', 'VVPR', 'AMPS',
        'BE', 'CLNE', 'FCEL', 'GEVO', 'AMTX', 'BLDP', 'PLUG', 'HYSR', 'AMRC', 'FLNC',
        'QS', 'SES', 'ABML', 'CBAT', 'POLA', 'EOSE', 'FREYR', 'ENVX', 'PCVX', 'STEM',
        'BLNK', 'CHPT', 'EVGo', 'DCFC', 'VLTA', 'WBX', 'ALPP', 'AYRO', 'NWTN', 'SOLO',
        'LEV', 'GOEV', 'ARVL', 'FSR', 'RIDE', 'WKHS', 'GEV', 'ACTC', 'NGAC', 'CCIV',
        'PSNY', 'REE', 'INDI', 'ELMS', 'HYZN', 'PTRA', 'MPAA', 'EMBK', 'XL', 'PROTERRA',
        'BIRD', 'HIMS', 'BROS', 'CANO', 'DNA', 'NTLA', 'BEAM', 'CRSP', 'EDIT', 'VERV',
        'MASS', 'VCYT', 'FATE', 'BLUE', 'QURE', 'SGMO', 'CRIS', 'VKTX', 'VERU', 'NRIX',
        'ALLO', 'ABUS', 'ADAP', 'ADMA', 'ADPT', 'ADTX', 'ADVM', 'AGLE', 'AGRX', 'AIMD',
        'AKBA', 'AKRO', 'ALDX', 'ALEC', 'ALIM', 'ALLO', 'ALNY', 'ALVR', 'AMED', 'AMGN',
        'AMPH', 'ANGO', 'ANIP', 'ANPC', 'ANTE', 'APDN', 'APTO', 'APYX', 'ARDX', 'ARDS',
        'ARQT', 'ARWR', 'ASLN', 'ASND', 'ASRT', 'ASXC', 'ATNF', 'ATOS', 'ATRA', 'ATNM',
        'AVDL', 'AVEO', 'AVGR', 'AVIR', 'AVRO', 'AVXL', 'AXGN', 'AXLA', 'AXNX', 'AXSM',
        'AYTU', 'BCAB', 'BCDA', 'BCEL', 'BCLI', 'BCRX', 'BDSX', 'BDTX', 'BEAT', 'BFRI',
        'BHTG', 'BIOL', 'BIOX', 'BLRX', 'BMEA', 'BMRA', 'BNGO', 'BOLD', 'BPTH', 'BRTX',
        'BSGM', 'BSQR', 'BTAI', 'BVXV', 'BYSI', 'BZUN', 'CAPR', 'CARA', 'CARV', 'CATB',
        'CBAY', 'CBIO', 'CBRX', 'CCCC', 'CCXI', 'CDAK', 'CDMO', 'CDNA', 'CDTX', 'CDXC',
        'CDXS', 'CEMI', 'CENT', 'CERE', 'CERS', 'CGEN', 'CGEM', 'CHEK', 'CHMA', 'CHRS',

        # Additional Russell 2000 - Small-Cap Tech (100 stocks)
        'AAOI', 'AAON', 'ABCL', 'ABEO', 'ACIA', 'ACMR', 'ACRS', 'ADTN', 'ADUS', 'AEIS',
        'AEYE', 'AFRM', 'AIRC', 'AKAM', 'ALKT', 'ALRM', 'ALTR', 'AMBA', 'AMED', 'AMKR',
        'AMSC', 'AMWD', 'ANGI', 'ANNX', 'AOSL', 'APPF', 'ARAY', 'ARCE', 'ARCT', 'ARLO',
        'ARVL', 'ASYS', 'ATEC', 'ATEX', 'ATOM', 'ATTO', 'AUDC', 'AVAV', 'AVNW', 'AXTI',
        'AZTA', 'BBAI', 'BBCP', 'BCOR', 'BCOV', 'BELFB', 'BGNE', 'BILL', 'BKKT', 'BLBD',
        'BLFS', 'BLNK', 'BMBL', 'BNFT', 'BNSO', 'BPMC', 'BPOP', 'BRID', 'BRKR', 'BRKS',
        'BTBT', 'BTCT', 'BWMX', 'BYFC', 'CALX', 'CAMP', 'CAMT', 'CARB', 'CART', 'CASS',
        'CCRN', 'CCSI', 'CDLX', 'CDNS', 'CDZI', 'CEVA', 'CGNT', 'CHRS', 'CIEN', 'CIFR',
        'CIGI', 'CGNX', 'CLBT', 'CLFD', 'CLIR', 'CLOU', 'CLPT', 'CLRO', 'CLVT', 'CLWT',
        'CMCT', 'CMPR', 'CNCE', 'CNMD', 'CNOB', 'CNSL', 'CNST', 'CNXC', 'CNXN', 'COGT',

        # Additional Russell 2000 - Small-Cap Healthcare (100 stocks)
        'CLDX', 'CMPS', 'CNTA', 'COCP', 'COHN', 'COHR', 'COLB', 'COLL', 'CORT', 'COYA',
        'CPRX', 'CRBP', 'CRDF', 'CRNX', 'CRSP', 'CRTX', 'CRVS', 'CRWS', 'CSTL', 'CTHR',
        'CTMX', 'CTIC', 'CTSO', 'CTRN', 'CTXR', 'CUTR', 'CVAC', 'CVCO', 'CVGW', 'CVIG',
        'CVLT', 'CVRX', 'CWST', 'CXDO', 'CYCN', 'CYRX', 'CYTK', 'CZNC', 'DADA', 'DAIO',
        'DAWN', 'DBRG', 'DCGO', 'DCOM', 'DENN', 'DERM', 'DFIN', 'DGII', 'DGLY', 'DIOD',
        'DJCO', 'DLHC', 'DMAC', 'DMRC', 'DNLI', 'DOGZ', 'DOMO', 'DORM', 'Doug', 'DRCT',
        'DRMA', 'DRRX', 'DSGX', 'DSWL', 'DTIL', 'DXPE', 'DXYN', 'DYAI', 'DYNC', 'EAGL',
        'EARN', 'EARS', 'EAST', 'EBIX', 'EBON', 'ECHO', 'ECOR', 'ECPG', 'EDAP', 'EDBL',
        'EDIT', 'EDRY', 'EDSA', 'EDUC', 'EEFT', 'EFOI', 'EFSC', 'EGAN', 'EGBN', 'EGLE',
        'EGRX', 'EH', 'EHTH', 'EIGR', 'EKSO', 'ELDN', 'ELIO', 'ELMD', 'ELSE', 'ELTK',

        # Additional Russell 2000 - Small-Cap Financials (75 stocks)
        'EMBC', 'EMCF', 'EME', 'EMKR', 'EML', 'EMMA', 'EMMS', 'EMP', 'ENCP', 'ENFN',
        'ENG', 'ENIA', 'ENJY', 'ENLV', 'ENOB', 'ENOV', 'ENR', 'ENSC', 'ENTA', 'ENTG',
        'ENVA', 'ENVB', 'ENVI', 'ENVX', 'ENZC', 'EOLS', 'EOSE', 'EPAC', 'EPAM', 'EPAY',
        'EPIX', 'EPRT', 'EQBK', 'EQIX', 'EQOS', 'EQRX', 'ERAS', 'ERES', 'ERIC', 'ERIE',
        'ERII', 'ERIEY', 'EROS', 'ESBK', 'ESCA', 'ESEA', 'ESGR', 'ESGRP', 'ESGV', 'ESLT',
        'ESMT', 'ESNT', 'ESOA', 'ESPN', 'ESPR', 'ESQ', 'ESSA', 'ESSC', 'ESTA', 'ESTC',
        'ESTY', 'ESXB', 'ETAO', 'ETD', 'ETNB', 'ETON', 'ETSY', 'ETTX', 'EUDA', 'EURN',
        'EUSA', 'EVCM', 'EVGN', 'EVGO', 'EVGR', 'EVLO', 'EVLV', 'EVOK', 'EVRG', 'EVRI',

        # Additional Russell 2000 - Small-Cap Consumer (75 stocks)
        'EVTV', 'EVTC', 'EWTX', 'EWZS', 'EXAS', 'EXEL', 'EXFY', 'EXLS', 'EXOD', 'EXPD',
        'EXPE', 'EXPI', 'EXPO', 'EXTR', 'EYE', 'EYEN', 'EYEG', 'EYES', 'EYESW', 'EYPT',
        'EZFL', 'EZGO', 'EZPW', 'FAAR', 'FAAS', 'FAF', 'FALC', 'FAMI', 'FANG', 'FANH',
        'FARM', 'FARO', 'FAST', 'FATBB', 'FATE', 'FATP', 'FBNC', 'FBIZ', 'FBLG', 'FBMS',
        'FBNK', 'FBRT', 'FBRX', 'FCAP', 'FCBC', 'FCBP', 'FCCO', 'FCCY', 'FCEL', 'FCFS',
        'FCNCA', 'FCPT', 'FDBC', 'FDMT', 'FDUS', 'FEIM', 'FELE', 'FELP', 'FENC', 'FEND',
        'FENG', 'FERG', 'FEXD', 'FFBC', 'FFBW', 'FFG', 'FFIC', 'FFIE', 'FFIN', 'FFIV',
        'FFNW', 'FFWM', 'FGBI', 'FGBIP', 'FGEN', 'FGF', 'FGFPP', 'FGI', 'FGIWW', 'FGMC',

        # Additional Russell 2000 - Small-Cap Industrials (75 stocks)
        'FHB', 'FHI', 'FHLT', 'FIAC', 'FIBK', 'FICO', 'FINV', 'FINW', 'FISI', 'FITB',
        'FITBI', 'FITBO', 'FITBP', 'FIVN', 'FIXD', 'FIXX', 'FIZZ', 'FKWL', 'FLAG', 'FLDM',
        'FLEX', 'FLFV', 'FLIC', 'FLKS', 'FLLC', 'FLLCU', 'FLME', 'FLMN', 'FLMNW', 'FLNG',
        'FLNT', 'FLUX', 'FLWS', 'FLXN', 'FLXS', 'FLYW', 'FMAO', 'FMBH', 'FMBI', 'FMC',
        'FMIV', 'FMNB', 'FN', 'FNA', 'FNCB', 'FNCH', 'FNHC', 'FNJN', 'FNKO', 'FNLC',
        'FNRN', 'FNVT', 'FNWB', 'FNWD', 'FOCS', 'FOE', 'FOLD', 'FOMX', 'FONE', 'FOR',
        'FORD', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOX', 'FOXA', 'FOXF', 'FOXW', 'FPAC',
        'FPAY', 'FPAYU', 'FPAYW', 'FPF', 'FPH', 'FPI', 'FRAF', 'FRBA', 'FRBK', 'FREE',

        # Additional Russell 2000 - Small-Cap Energy & Materials (65 stocks)
        'FREQ', 'FRES', 'FRGE', 'FRGAP', 'FRGT', 'FRHC', 'FRLA', 'FRLAU', 'FRLAW', 'FRM',
        'FRME', 'FRMEP', 'FRO', 'FRPH', 'FRPT', 'FRSX', 'FRZA', 'FSBC', 'FSBW', 'FSCO',
        'FSD', 'FSEA', 'FSFG', 'FSK', 'FSLR', 'FSLY', 'FSM', 'FSNB', 'FSP', 'FSR',
        'FSS', 'FSTR', 'FSV', 'FTAA', 'FTAAU', 'FTAAW', 'FTAI', 'FTCH', 'FTCI', 'FTCV',
        'FTCVU', 'FTCVW', 'FTDR', 'FTEK', 'FTF', 'FTFT', 'FTHM', 'FTHY', 'FTI', 'FTK',
        'FTLF', 'FTNT', 'FTRE', 'FTRP', 'FTSH', 'FTV', 'FUBO', 'FUEL', 'FUL', 'FULC',
        'FULT', 'FULTP', 'FUN', 'FUNC', 'FUND', 'FURY', 'FUSB', 'FUTU', 'FVCB', 'FVE',

        # Additional Russell 2000 - Micro-Cap Diversified (125 stocks)
        'FWAC', 'FWBI', 'FWONA', 'FWONK', 'FWRD', 'FXCO', 'FXLV', 'FXNC', 'FYLD', 'GABC',
        'GAIA', 'GAIN', 'GAINL', 'GAINM', 'GAINN', 'GAINO', 'GALT', 'GAM', 'GAMB', 'GAMC',
        'GAME', 'GAN', 'GANX', 'GASS', 'GATE', 'GATEU', 'GATEW', 'GBAB', 'GBCI', 'GBDC',
        'GBIO', 'GBLI', 'GBLIL', 'GBLIZ', 'GBNK', 'GBNY', 'GBRGR', 'GBRGU', 'GBRGW', 'GBS',
        'GBT', 'GCBC', 'GCMG', 'GCMGW', 'GCO', 'GCOR', 'GCP', 'GCT', 'GCTK', 'GCV',
        'GDEV', 'GDEVW', 'GDHG', 'GDNR', 'GDNRW', 'GDO', 'GDOC', 'GDOT', 'GDRX', 'GDS',
        'GDST', 'GDTC', 'GDYN', 'GDYNW', 'GECC', 'GECCM', 'GECCN', 'GECCO', 'GEF', 'GEFA',
        'GEG', 'GEHC', 'GEL', 'GEN', 'GENC', 'GENE', 'GENI', 'GENK', 'GEOS', 'GERN',
        'GES', 'GESCO', 'GETD', 'GETY', 'GEVO', 'GFAI', 'GFF', 'GFGD', 'GFGF', 'GFI',
        'GFL', 'GFOR', 'GFVD', 'GFX', 'GGAA', 'GGAAU', 'GGAAW', 'GGAL', 'GGE', 'GGMC',
        'GGMCU', 'GGMCW', 'GGN', 'GGOOW', 'GGR', 'GGROW', 'GGZ', 'GH', 'GHC', 'GHIX',
        'GHIXU', 'GHIXW', 'GHL', 'GHLD', 'GHRS', 'GHSI', 'GIAC', 'GIACU', 'GIACW', 'GIB',
        'GIFI', 'GIFT', 'GIII', 'GIIX', 'GIIXU', 'GIIXW', 'GIL',

        # Final Russell 2000 Additions - Mixed Sectors (100 stocks)
        'GILD', 'GILT', 'GILTI', 'GIS', 'GKOS', 'GL', 'GLAD', 'GLADD', 'GLBE', 'GLBS',
        'GLBZ', 'GLDD', 'GLDI', 'GLG', 'GLHA', 'GLHAU', 'GLHAW', 'GLMD', 'GLNG', 'GLOP',
        'GLOV', 'GLP', 'GLPG', 'GLPI', 'GLRE', 'GLSI', 'GLST', 'GLT', 'GLTO', 'GLUE',
        'GLXG', 'GLYC', 'GM', 'GMAB', 'GMBL', 'GMBLW', 'GMDA', 'GME', 'GMED', 'GMGI',
        'GMLP', 'GMLPP', 'GMRE', 'GMS', 'GMTX', 'GNCA', 'GNFT', 'GNK', 'GNL', 'GNLX',
        'GNMA', 'GNPX', 'GNRC', 'GNSS', 'GNTA', 'GNTX', 'GNTY', 'GNUS', 'GO', 'GOCCU',
        'GOCCW', 'GODN', 'GOEV', 'GOEVW', 'GOF', 'GOGL', 'GOGO', 'GOL', 'GOLD', 'GOLF',
        'GOOS', 'GORO', 'GORV', 'GOSS', 'GOTU', 'GOVX', 'GOVXW', 'GP', 'GPAC', 'GPACU',
        'GPACW', 'GPC', 'GPI', 'GPJA', 'GPK', 'GPL', 'GPMT', 'GPN', 'GPOR', 'GPRE',
        'GPRK', 'GPRO', 'GPS', 'GPTX', 'GPX', 'GRAB', 'GRABW', 'GRAL', 'GRBK', 'GRC',

        # ==================== INTERNATIONAL STOCKS ====================

        # Europe - Technology & Semiconductors (40 stocks)
        'ASML', 'SAP', 'SHOP', 'SE', 'SPOT', 'NICE', 'CYBR', 'CHKP', 'WIX', 'MNDY',
        'STM', 'ERIC', 'NOK', 'ARM', 'INFN', 'LITE', 'SWKS', 'SMCI', 'LOGI', 'LSCC',
        'SGMS', 'SSNLF', 'IFNNY', 'NOKIA', 'EADSY', 'SIEGY', 'TKOMY', 'TSM', 'UMC', 'ASX',
        'HIMX', 'SPIL', 'OLED', 'AU', 'HTHT', 'VNET', 'KC', 'TIGR', 'FUTU', 'UP',

        # Europe - Financials & Banking (50 stocks)
        'BCS', 'DB', 'CS', 'UBS', 'BBVA', 'SAN', 'INGA', 'ING', 'BNP', 'AXA',
        'SCOR', 'AEGN', 'AEGON', 'NN', 'ASR', 'CRDI', 'UCG', 'ISP', 'BPSO', 'BAMI',
        'BAMI', 'CABK', 'RBS', 'LYG', 'HSBC', 'VOD', 'BT', 'TEF', 'TI', 'ORAN',
        'FTE', 'VIV', 'EQNR', 'NG', 'SHEL', 'BP', 'TTE', 'RDSB', 'RDS.B', 'TOT',
        'ENI', 'REP', 'REPYY', 'GALP', 'OMV', 'PKN', 'LONN', 'MOWI', 'BAKKA', 'DNO',

        # Europe - Consumer & Retail (50 stocks)
        'NVO', 'NOVO', 'NOVOB', 'AZN', 'GSK', 'SNY', 'BAYRY', 'RHHBY', 'BAYN', 'NVS',
        'ROG', 'NESN', 'ADDYY', 'OR', 'MC', 'RMS', 'KER', 'CFR', 'BURBY', 'IDEXY',
        'HENKY', 'HEN3', 'BEI', 'LRLCY', 'LRLCF', 'LVMUY', 'PRDSY', 'LWAY', 'WPP', 'PUBGY',
        'REXR', 'FP', 'EDF', 'EDP', 'IBE', 'ELE', 'ENGI', 'VIE', 'EOAN', 'RWE',
        'E.ON', 'STLAM', 'STOXX', 'NOKIA', 'FIAT', 'STLA', 'RACE', 'VOW3', 'BMW', 'DAI',

        # Asia-Pacific - China (60 stocks)
        'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TCOM', 'BILI', 'IQ', 'TME', 'HUYA',
        'DOYU', 'MOMO', 'YY', 'JMIA', 'WB', 'VIPS', 'ATHM', 'BZUN', 'DADA', 'DDL',
        'DAO', 'BEST', 'TUYA', 'RLX', 'GOTU', 'EDU', 'TAL', 'GOTU', 'DUO', 'LAIX',
        'TWOU', 'ZYXI', 'TEDU', 'YANG', 'YINN', 'KWEB', 'CQQQ', 'GXC', 'FXI', 'MCHI',
        'NIO', 'XPEV', 'LI', 'KNDI', 'NIU', 'SOLO', 'WKHS', 'IDEX', 'BLNK', 'SBE',
        'QS', 'GOEV', 'AYRO', 'KANDI', 'NKLA', 'RIDE', 'HYLN', 'CIIC', 'SBE', 'PSNY',

        # Asia-Pacific - India (40 stocks)
        'INFY', 'WIT', 'HDB', 'IBN', 'SIFY', 'REDF', 'REDY', 'TTM', 'VEDL', 'WNS',
        'YTRA', 'RDY', 'ICICI', 'AXISB', 'SBIN', 'HDFCB', 'KOTAKB', 'YESBK', 'PNB', 'BOB',
        'CANBK', 'IDBI', 'UNBNK', 'INDBNK', 'FED', 'DHFL', 'ING', 'PIN', 'ITC', 'HIND',
        'TATA', 'RLNC', 'BHARAT', 'ONGC', 'COAL', 'NTPC', 'POWERGD', 'IOC', 'BPCL', 'HPCL',

        # Asia-Pacific - South Korea (35 stocks)
        'TSM', 'SSNLF', 'LPL', 'SKM', 'KB', 'SHG', 'HMC', 'TM', 'PCRFY', 'HYMTF',
        'SMSN', 'LG', 'LGIH', 'LGLG', 'LGEL', 'LGCL', 'LGLD', 'HYSN', 'KIMTF', 'SSNGY',
        'KEP', 'PKX', 'SSL', 'SPOT', 'HYUD', 'KIA', 'VLKAF', 'SMAWF', 'POAHY', 'NCTY',
        'NAVER', 'KAKAO', 'COUPN', 'TCEHY', 'BABA',

        # Asia-Pacific - Japan (50 stocks)
        'SONY', 'TM', 'HMC', 'NSANY', 'NTDOY', 'FUJIY', 'HTHIY', 'SNEJF', 'MSBHF', 'MITSY',
        'MITSF', 'SMFG', 'MTU', 'MFG', 'MUFG', 'NMR', 'KB', 'SMFNF', 'MITSUBISHI', 'ITOCHU',
        'MARUY', 'SOMMY', 'CANNY', 'KDDIY', 'TOELY', 'FANUY', 'PCRFY', 'FUJHD', 'RICOY', 'SEKEY',
        'SHKLY', 'DNZOY', 'SZKMY', 'KAISY', 'AJINY', 'OLIMP', 'NPSNY', 'SHCAY', 'TAKAY', 'SYIEY',
        'AIQUY', 'DSKYY', 'OTSKY', 'SXRCY', 'RCRUY', 'KUBTY', 'YAMCY', 'MZDAY', 'DPSGY', 'KGFHY',

        # Latin America (40 stocks)
        'VALE', 'PBR', 'ITUB', 'BBD', 'ABEV', 'SBS', 'BSAC', 'BVN', 'GGAL', 'YPF',
        'TEO', 'TX', 'CIG', 'PAM', 'SID', 'CBD', 'ERJ', 'GOL', 'CIB', 'BSBR',
        'EBR', 'ELET', 'VIV', 'AMX', 'TV', 'TSU', 'TIMB', 'FMX', 'KOF', 'AC',
        'ASUR', 'GAP', 'OMA', 'PAC', 'VLRS', 'SU', 'QIWI', 'MAIL', 'YNDX', 'OZON',

        # Middle East & Africa (25 stocks)
        'TEVA', 'CHKP', 'CYBR', 'NICE', 'WIX', 'MNDY', 'FVRR', 'LMND', 'GLBE', 'MGIC',
        'TIGO', 'MTN', 'SBSA', 'JSE', 'IMPUY', 'ANGPY', 'SBSW', 'HGTY', 'GOLD', 'AU',
        'GFI', 'HMY', 'RGLD', 'NG', 'DRIP',

        # Canada (40 stocks)
        'SHOP', 'TD', 'RY', 'BNS', 'BMO', 'CM', 'ENB', 'CNQ', 'TRP', 'SU',
        'CNR', 'CP', 'ABX', 'GOLD', 'NEM', 'AEM', 'FNV', 'WPM', 'PAAS', 'EGO',
        'BB', 'LSPD', 'REAL', 'WELL', 'DOC', 'FOOD', 'QSR', 'RBI', 'MGA', 'ATD',
        'WCN', 'BEP', 'BEPC', 'AQN', 'HASI', 'NPI', 'BLX', 'BAM', 'BIP', 'BIPC',

        # Australia & New Zealand (30 stocks)
        'BHP', 'RIO', 'WES', 'CSL', 'CBA', 'NAB', 'ANZ', 'WBC', 'MQG', 'TLS',
        'WOW', 'WPL', 'STO', 'ORG', 'S32', 'FMG', 'NCM', 'EVN', 'NST', 'SFR',
        'RMD', 'COH', 'REA', 'SEK', 'XRO', 'APT', 'WTC', 'A2M', 'TWE', 'ALU',

        # Emerging Markets - Southeast Asia (35 stocks)
        'GRAB', 'SEA', 'BABA', 'BEKE', 'DIDI', 'TME', 'BGNE', 'VIPS', 'ZTO', 'YMM',
        'GDS', 'HTHT', 'IQ', 'BIDU', 'KC', 'GOTU', 'TAL', 'EDU', 'BEKE', 'DOYU',
        'HUYA', 'YY', 'MOMO', 'WB', 'BEST', 'TUYA', 'DAO', 'DADA', 'DDL', 'LU',
        'RLX', 'MOGU', 'TIGR', 'FUTU', 'UP',

        # Global ADRs - Telecommunications (25 stocks)
        'VOD', 'TEF', 'TI', 'VIV', 'ORAN', 'FTE', 'AMX', 'CHT', 'CHL', 'TU',
        'SKM', 'DCM', 'PHI', 'TEO', 'TIM', 'VIP', 'VIVHY', 'DTEGY', 'TMOBY', 'NCDX',
        'TELF', 'TLSNF', 'TLSN', 'TLSYY', 'BCE',

        # Global ADRs - Energy & Utilities (35 stocks)
        'EQNR', 'E', 'SHEL', 'BP', 'TTE', 'TOT', 'SU', 'CNQ', 'IMO', 'CVE',
        'ENB', 'TRP', 'PBA', 'EC', 'ENIC', 'EOAN', 'RWE', 'IBE', 'ELE', 'EDP',
        'ENGI', 'FP', 'NG', 'VEDL', 'SCCO', 'FCX', 'TECK', 'HBM', 'GLEN', 'AAL',
        'VALE', 'RIO', 'BHP', 'SSRM', 'PAAS',

        # Global ADRs - Industrials & Conglomerates (30 stocks)
        'UL', 'ULVR', 'UN', 'DEO', 'DANOY', 'NSRGY', 'UNLRY', 'BUD', 'SAM', 'TAP',
        'STZ', 'HEINY', 'CCEP', 'KO', 'PEP', 'SBMRY', 'SAPMY', 'SDMRY', 'BASFY', 'BAYRY',
        'LNVGY', 'LNVGF', 'LIN', 'AIR', 'EADSY', 'BA', 'RTX', 'GD', 'LMT', 'NOC'
    ]
