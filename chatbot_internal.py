import getpass
import os
import pandas as pd
import os
import requests
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import asyncio
import json
from typing import List, Dict, Set
from dataclasses import dataclass, field, asdict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class ProjectInfo:
    name: str
    technologies: Set[str] = field(default_factory=set)
    description: str = ""
    developers: Set[str] = field(default_factory=set)
    platforms: Set[str] = field(default_factory=set)
    file_source: str = ""
    page_numbers: Set[int] = field(default_factory=set)
    relevance_score: float = 0.0  # Added for ranking

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o")

OUTPUT_FOLDER = 'pdfs'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

df = pd.read_excel('SalesData.xlsx')

for index, row in tqdm(df.iterrows(), total=len(df), desc='Downloading PDFs'):
    name = row['Name of Project']
    drive_url = row['Drive Link']
    pdf_filename = f'{name}.pdf'
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

    # Skip if the file is already downloaded
    if os.path.exists(pdf_path):
        print(f"{pdf_filename} already exists. Skipping download.")
        continue

    # Check if drive_url is valid
    if not drive_url or not drive_url.startswith('https://drive.google.com/'):
        print(f"Invalid drive_url: {drive_url} for {name}")
        continue

    # Extract file_id
    url_parts = drive_url.split('/')
    if len(url_parts) < 6:  # Expected format: https://drive.google.com/file/d/{file_id}/...
        print(f"Unexpected drive_url format: {drive_url} for {name}")
        continue

    file_id = url_parts[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}&export=download'

    try:
        response = requests.get(download_url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the downloaded file
        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        print(f'Downloaded {pdf_filename}')
    except requests.RequestException as e:
        print(f'Error downloading {pdf_filename}: {e}')
    except Exception as e:
        print(f'Error saving {pdf_filename}: {e}')


class PersistentProjectProcessor:
    def __init__(self, folder_path: str, cache_dir: str = "project_cache"):
        self.folder_path = folder_path
        self.cache_dir = cache_dir
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            chunk_size=1000
        )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.vector_store = None
        self.projects = {}
        self.technology_index = {}
        self.last_processed_time = None

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # File paths for cached data
        self.cache_paths = {
            'projects': os.path.join(cache_dir, 'projects.json'),
            'tech_index': os.path.join(cache_dir, 'tech_index.json'),
            'vector_store': os.path.join(cache_dir, 'vector_store.faiss'),
            'metadata': os.path.join(cache_dir, 'metadata.json')
        }

        # Enhanced technology patterns with more variations
        self.tech_patterns = {
            'frontend': r'\b(React|Angular|Vue|Next\.js|Svelte|jQuery|HTML5|CSS3|Bootstrap|Tailwind|Material-UI)\b',
            'backend': r'\b(Node\.js|Express|Django|Flask|Spring|Laravel|FastAPI|Rails|ASP\.NET|PHP)\b',
            'database': r'\b(MongoDB|PostgreSQL|MySQL|Redis|Cassandra|SQLite|Oracle|DynamoDB|Firebase|ElasticSearch)\b',
            'mobile': r'\b(iOS|Android|React Native|Flutter|Swift|Kotlin|Xamarin|Ionic|PWA|Mobile Web)\b',
            'devops': r'\b(Docker|Kubernetes|Jenkins|AWS|Azure|GCP|CircleCI|Travis|GitLab|Terraform)\b',
            'languages': r'\b(Python|JavaScript|TypeScript|Java|C\+\+|Go|Rust|Ruby|C#|Scala)\b'
        }

    async def extract_project_info(self, text: str, filename: str, page_num: int) -> None:
        """Enhanced project information extraction"""
        # Expanded project patterns
        project_patterns = [
            r"Project\s*:\s*([^\n.]+)",
            r"Project Name\s*:\s*([^\n.]+)",
            r"Title\s*:\s*([^\n.]+)(?=\s*Project)",
            r"Application Name\s*:\s*([^\n.]+)",
            r"System Name\s*:\s*([^\n.]+)",
            r"Product Name\s*:\s*([^\n.]+)"
        ]

        for pattern in project_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                project_name = match.group(1).strip()
                if project_name not in self.projects:
                    self.projects[project_name] = ProjectInfo(
                        name=project_name,
                        file_source=filename
                    )
                project_info = self.projects[project_name]
                project_info.page_numbers.add(page_num)

                # Enhanced technology extraction
                for category, pattern in self.tech_patterns.items():
                    for tech in re.finditer(pattern, text, re.IGNORECASE):
                        tech_name = tech.group(0)
                        project_info.technologies.add(tech_name)
                        if tech_name not in self.technology_index:
                            self.technology_index[tech_name] = set()
                        self.technology_index[tech_name].add(project_name)

                # Enhanced description extraction with multiple patterns
                description_patterns = [
                    r"Description\s*:\s*([^\n.]+)",
                    r"Overview\s*:\s*([^\n.]+)",
                    r"Summary\s*:\s*([^\n.]+)",
                    r"About\s*:\s*([^\n.]+)"
                ]
                for desc_pattern in description_patterns:
                    if desc_match := re.search(desc_pattern, text, re.IGNORECASE):
                        project_info.description = desc_match.group(1).strip()
                        break

                # Enhanced developer/role extraction
                role_patterns = [
                    r"Developer[s]?\s*:\s*([^\n.]+)",
                    r"Team Member[s]?\s*:\s*([^\n.]+)",
                    r"Contributors?\s*:\s*([^\n.]+)",
                    r"Created by\s*:\s*([^\n.]+)"
                ]
                for role_pattern in role_patterns:
                    if role_match := re.findall(role_pattern, text, re.IGNORECASE):
                        project_info.developers.update(
                            dev.strip() for dev in role_match[0].split(',')
                        )

    def find_relevant_projects(self, query: str) -> List[Dict]:
        """Find all relevant projects without any limit"""
        scored_projects = []
        query_lower = query.lower()

        # Score and collect ALL matching projects
        for project_name, project in self.projects.items():
            score = 0.0

            # Match technologies
            for category, pattern in self.tech_patterns.items():
                techs = re.finditer(pattern, query, re.IGNORECASE)
                for tech in techs:
                    tech_name = tech.group(0)
                    if tech_name.lower() in [t.lower() for t in project.technologies]:
                        score += 2.0

            # Match project name
            if query_lower in project_name.lower():
                score += 1.5

            # Match description
            if project.description and query_lower in project.description.lower():
                score += 1.0

            # Include ALL projects with any relevance
            if score > 0:
                project.relevance_score = score
                scored_projects.append({
                    "name": project.name,
                    "technologies": list(project.technologies),
                    "description": project.description,
                    "developers": list(project.developers),
                    "file": project.file_source,
                    "pages": list(project.page_numbers),
                    "score": score
                })

        # Sort by score but return ALL results
        return sorted(scored_projects, key=lambda x: x['score'], reverse=True)

    async def query_projects(self, query: str) -> str:
        """Query with unlimited results"""
        if not self.vector_store:
            return "Please process PDFs first using process_pdfs()"

        # Get ALL relevant projects
        relevant_projects = self.find_relevant_projects(query)

        # Get more context from vector search
        docs = self.vector_store.similarity_search(query, k=40)  # Increased for more context
        context = "\n".join(doc.page_content for doc in docs)

        # Modified prompt to return JSON
        prompt = PromptTemplate(
            template="""
            Based on the following information, please provide a comprehensive answer about ALL relevant projects.

            Context from documents:
            {context}

            Relevant projects found (ranked by relevance):
            {projects}

            Question: {question}

            Important: Please include EVERY project that matches the query criteria, no matter how many there are.
            For each project, provide:
            - Project name
            - Technologies used
            - Developer roles (if available)
            - Brief description
            - Source file and page numbers

            Return the result in the following JSON format:
            [
                {{
                    "name": "Project Name",
                    "technologies": ["Technology1", "Technology2", ...],
                    "developers": ["Developer1", "Developer2", ...],
                    "description": "Brief description of the project",
                    "file": "Source file name",
                    "pages": [Page numbers where the project is found],
                    "score": Relevance score
                }},
                ...
            ]

            If no matching projects are found, return the following JSON:
            {{
                "error": "No matching projects were found."
            }}

            Answer:
            """,
            input_variables=["context", "projects", "question"]
        )

        chain = (
            {
                "context": lambda _: context,
                "projects": lambda _: str(relevant_projects),
                "question": lambda _: query
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        raw_res = await chain.ainvoke("")
        result = await self.format_and_fix_json(raw_res)
        
        return result
    
    async def clean_and_parse_json(self, input_str):
        # Remove markdown code block indicators and extra content
        cleaned_input = re.sub(r'^```json?\s*', '', input_str, flags=re.MULTILINE)
        cleaned_input = re.sub(r'```\s*$', '', cleaned_input, flags=re.MULTILINE)

        try:
            # Parse the cleaned JSON string
            parsed_data = json.loads(cleaned_input)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    
    async def format_and_fix_json(self, json_string: str):
        # Replace any placeholder or empty fields with dummy values
        if not json_string.strip():
            # If the input string is empty, we can return an empty list or appropriate default
            json_string = "[]"
        
        # Replace placeholders with valid dummy values
        json_string = json_string.replace("Page numbers where the project is found", "")
        json_string = json_string.replace("Relevance score", "0")
        
        # Replace empty strings or missing values with default dummy values
        json_string = json_string.replace('""', '"dummy_value"')
        
        try:
            # Try parsing the updated JSON
            print(json_string)
            cleaned_input = await self.clean_and_parse_json(json_string)
            print(cleaned_input)
            
            # Ensure that the parsed JSON follows the correct structure
            if not isinstance(cleaned_input, list):
                raise ValueError("Expected a list of projects in the JSON")
            
            for item in cleaned_input:
                if not isinstance(item, dict):
                    raise ValueError(f"Each item in the JSON array must be an object: {item}")
                
                # Ensure necessary keys exist and replace missing fields with default dummy values
                item.setdefault("pages", [])
                item.setdefault("score", 0)
                item.setdefault("name", "Unknown Project")
                item.setdefault("technologies", ["Unknown Technology"])
                item.setdefault("developers", [])
                item.setdefault("description", "No description available.")
                item.setdefault("file", "Unknown file")

            print(cleaned_input)
            return cleaned_input
        
        except json.JSONDecodeError as e:
            # Handle and report errors in decoding
            raise ValueError(f"Error decoding JSON: {e}")

    async def process_single_pdf(self, filename: str) -> List[str]:
        """Process a single PDF file"""
        file_path = os.path.join(self.folder_path, filename)
        chunks = []

        try:
            loader = PyPDFLoader(file_path)
            pages = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )

            # Use CharacterTextSplitter for faster splitting
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )

            for i, page in enumerate(pages):
                # Clean text
                text = re.sub(r'\s+', ' ', page.page_content).strip()
                # Extract information
                await self.extract_project_info(text, filename, i + 1)
                # Split into chunks
                page_chunks = splitter.split_text(text)
                chunks.extend(page_chunks)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

        return chunks

    async def process_pdfs(self):
        """Process all PDFs in parallel"""
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]

        # Process PDFs in parallel
        tasks = [self.process_single_pdf(pdf) for pdf in pdf_files]
        chunks_list = await asyncio.gather(*tasks)

        # Flatten chunks list
        all_chunks = [chunk for chunks in chunks_list for chunk in chunks]

        # Create FAISS index
        texts = [{"content": chunk, "source": "pdf"} for chunk in all_chunks]
        self.vector_store = FAISS.from_texts(
            [text["content"] for text in texts],
            self.embeddings,
            metadatas=texts
        )

    def _should_reprocess(self) -> bool:
        """Check if PDFs need reprocessing by comparing modification times"""
        if not os.path.exists(self.cache_paths['metadata']):
            return True

        try:
            with open(self.cache_paths['metadata'], 'r') as f:
                metadata = json.load(f)
                cached_times = metadata.get('pdf_times', {})
                current_times = self._get_pdf_modification_times()

                return any(
                    current_times.get(file, 0) != cached_times.get(file, -1)
                    for file in set(current_times) | set(cached_times)
                )
        except:
            return True

    async def load_cached_data(self) -> bool:
        """Load processed data from cache if available and valid"""
        try:
            if not self._should_reprocess():
                # Load projects data
                with open(self.cache_paths['projects'], 'r') as f:
                    projects_data = json.load(f)
                    self.projects = {
                        name: ProjectInfo(
                            name=name,
                            technologies=set(data['technologies']),
                            description=data['description'],
                            developers=set(data['developers']),
                            platforms=set(data['platforms']),
                            file_source=data['file_source'],
                            page_numbers=set(data['page_numbers'])
                        )
                        for name, data in projects_data.items()
                    }

                # Load technology index
                with open(self.cache_paths['tech_index'], 'r') as f:
                    tech_data = json.load(f)
                    self.technology_index = {
                        tech: set(projects)
                        for tech, projects in tech_data.items()
                    }

                # Load vector store
                if os.path.exists(self.cache_paths['vector_store']):
                    self.vector_store = FAISS.load_local(
                        self.cache_paths['vector_store'],
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )

                print("Successfully loaded cached data!")
                return True

        except Exception as e:
            print(f"Error loading cached data: {str(e)}")

        return False

    def _get_pdf_modification_times(self) -> Dict[str, float]:
        """Get modification times of all PDFs in the folder"""
        pdf_times = {}
        for file in os.listdir(self.folder_path):
            if file.lower().endswith('.pdf'):
                path = os.path.join(self.folder_path, file)
                pdf_times[file] = os.path.getmtime(path)
        return pdf_times

    async def save_cached_data(self):
        """Save processed data to cache"""
        try:
            # Save projects data
            projects_data = {
                name: {
                    'technologies': list(info.technologies),
                    'description': info.description,
                    'developers': list(info.developers),
                    'platforms': list(info.platforms),
                    'file_source': info.file_source,
                    'page_numbers': list(info.page_numbers)
                }
                for name, info in self.projects.items()
            }
            with open(self.cache_paths['projects'], 'w') as f:
                json.dump(projects_data, f)

            # Save technology index
            tech_data = {
                tech: list(projects)
                for tech, projects in self.technology_index.items()
            }
            with open(self.cache_paths['tech_index'], 'w') as f:
                json.dump(tech_data, f)

            # Save vector store
            if self.vector_store:
                self.vector_store.save_local(self.cache_paths['vector_store'])

            # Save metadata
            metadata = {
                'pdf_times': self._get_pdf_modification_times(),
                'last_processed': datetime.now().isoformat()
            }
            with open(self.cache_paths['metadata'], 'w') as f:
                json.dump(metadata, f)

            print("Successfully saved processed data to cache!")

        except Exception as e:
            print(f"Error saving cached data: {str(e)}")

    async def initialize(self):
      """Initialize the processor - either load cached data or process PDFs"""
      if not await self.load_cached_data():
          print("Processing PDFs...")
          await self.process_pdfs()
          await self.save_cached_data()
          print("Processing and caching complete!")
