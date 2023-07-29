from ydata_profiling import ProfileReport


def auto_eda(df, title):
    profile = ProfileReport(df, title=title)
    profile.to_file(output_file='../reports/report.html')