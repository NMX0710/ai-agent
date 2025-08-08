from app.tools.file_operation_tool import read_file, write_file
from app.tools.pdf_generation_tool import generate_pdf
from app.tools.web_search_tool import web_search
from app.tools.web_scraping_tool import scrape_web_page
from app.tools.resource_download_tool import download_resource
from app.tools.terminal_operation_tool import execute_terminal_command

ALL_TOOLS = [
    read_file,
    write_file,
    generate_pdf,
    web_search,
    scrape_web_page,
    download_resource,
    execute_terminal_command,
]
