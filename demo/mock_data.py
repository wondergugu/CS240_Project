"""Generate synthetic retrieval data mimicking realistic RAG scenarios.

Simulates Wikipedia-style passages (100-200 tokens each) with realistic
redundancy patterns among top-ranked documents.
"""

import random

from srag.types import Document, Query


# Realistic-length passages about climate change (each ~100-200 words)
PASSAGES = [
    (
        "Climate change refers to long-term shifts in global temperatures and weather patterns. "
        "While natural factors like volcanic eruptions and solar cycles have historically influenced "
        "climate, human activities have been the primary driver since the industrial revolution. "
        "The burning of fossil fuels including coal, petroleum, and natural gas releases massive "
        "quantities of carbon dioxide and other greenhouse gases into the atmosphere. These gases "
        "trap heat from the sun, creating what scientists call the enhanced greenhouse effect. "
        "Since 1850, global average temperatures have risen by approximately 1.1 degrees Celsius, "
        "with most of this warming occurring in the past fifty years. The Intergovernmental Panel "
        "on Climate Change has confirmed with high confidence that human influence is the dominant "
        "cause of the observed warming trend."
    ),
    (
        "The greenhouse effect is a natural process that warms the Earth surface. When the sun "
        "energy reaches the Earth atmosphere, some of it is reflected back to space and the rest "
        "is absorbed and re-radiated by greenhouse gases. These gases include carbon dioxide, "
        "methane, nitrous oxide, and water vapor. The enhanced greenhouse effect occurs when "
        "human activities increase the concentration of these gases beyond natural levels. Carbon "
        "dioxide levels have risen from 280 parts per million before industrialization to over "
        "420 parts per million today. This unprecedented increase traps additional heat in the "
        "atmosphere, leading to global warming, rising sea levels, and disruption of weather "
        "patterns across the planet. Scientists use ice core data and atmospheric measurements "
        "to track these changes over thousands of years."
    ),
    (
        "Carbon dioxide emissions from industrial activities represent the largest single "
        "contributor to anthropogenic climate change. Power generation from coal and natural gas "
        "accounts for approximately 25 percent of global emissions. Transportation including cars "
        "trucks ships and aircraft contributes another 16 percent. Industrial processes such as "
        "cement production steel manufacturing and chemical processing add roughly 21 percent. "
        "Since the industrial revolution began in the late 18th century, cumulative CO2 emissions "
        "have exceeded 2400 gigatons. The rate of emissions has accelerated dramatically, with "
        "more CO2 released in the past 30 years than in the entire previous century. Developing "
        "nations are now the fastest growing sources of emissions as they industrialize."
    ),
    (
        "Deforestation and land use changes contribute significantly to climate change through "
        "multiple mechanisms. Forests act as carbon sinks, absorbing approximately 2.6 billion "
        "tons of CO2 annually. When trees are cut down or burned, the stored carbon is released "
        "back into the atmosphere. Tropical deforestation alone accounts for 8 to 10 percent of "
        "global greenhouse gas emissions. The Amazon rainforest, which stores an estimated 150 "
        "to 200 billion tons of carbon, has lost approximately 17 percent of its forest cover "
        "in the past 50 years. Agricultural expansion, logging, and urban development are the "
        "primary drivers. Additionally, degraded forests lose their ability to regulate local "
        "rainfall patterns, creating feedback loops that further accelerate forest loss."
    ),
    (
        "Rising global temperatures are causing polar ice caps and glaciers to melt at "
        "unprecedented rates. The Arctic has lost approximately 13 percent of its sea ice per "
        "decade since satellite measurements began in 1979. The Greenland ice sheet is losing "
        "about 280 billion tons of ice per year, while Antarctica loses roughly 150 billion tons "
        "annually. This melting contributes directly to sea level rise, which has accelerated to "
        "3.6 millimeters per year. Projections suggest sea levels could rise by 0.3 to 1.0 meters "
        "by 2100 depending on emission scenarios. Coastal communities housing hundreds of millions "
        "of people face increasing risks from flooding, storm surges, and saltwater intrusion "
        "into freshwater aquifers."
    ),
    (
        "Extreme weather events are becoming more frequent and intense due to climate change. "
        "Warmer ocean temperatures fuel stronger hurricanes and typhoons with higher wind speeds "
        "and heavier rainfall. Heat waves have become longer and more severe, with deadly events "
        "occurring more regularly across Europe, Asia, and North America. Drought conditions are "
        "expanding in subtropical regions, threatening water supplies and agricultural production. "
        "Conversely, heavy precipitation events have increased in many areas, leading to more "
        "frequent flooding. The economic damages from weather-related disasters have increased "
        "fivefold over the past 50 years, reaching hundreds of billions of dollars annually. "
        "These events disproportionately affect vulnerable populations in developing countries."
    ),
    (
        "The Paris Agreement adopted in 2015 represents the most comprehensive international "
        "effort to address climate change. Under this framework, 196 nations committed to "
        "limiting global warming to well below 2 degrees Celsius above pre-industrial levels, "
        "with efforts to limit the increase to 1.5 degrees. Each country submits nationally "
        "determined contributions outlining their emission reduction targets. The agreement also "
        "established mechanisms for climate finance, technology transfer, and capacity building "
        "to support developing nations. However, current pledges are insufficient to meet the "
        "stated goals, with projections showing warming of 2.4 to 2.8 degrees by 2100 under "
        "existing commitments. Regular stocktaking processes aim to ratchet up ambition over time."
    ),
    (
        "Renewable energy sources including solar photovoltaic, wind power, and hydroelectricity "
        "offer pathways to decarbonize the global energy system. Solar energy costs have fallen "
        "by 89 percent since 2010, making it the cheapest source of new electricity in most "
        "regions. Wind power capacity has grown by an average of 12 percent annually over the "
        "past decade. Together, renewables now account for approximately 30 percent of global "
        "electricity generation. Energy storage technologies, particularly lithium-ion batteries, "
        "are addressing the intermittency challenge. However, the transition requires massive "
        "investment in grid infrastructure, workforce retraining, and addressing the environmental "
        "impacts of mining materials needed for clean energy technologies."
    ),
    (
        "Ocean acidification represents a critical but often overlooked consequence of rising "
        "CO2 levels. The oceans absorb approximately 30 percent of atmospheric carbon dioxide, "
        "which reacts with seawater to form carbonic acid. Since pre-industrial times, ocean pH "
        "has decreased by 0.1 units, representing a 26 percent increase in acidity. This change "
        "threatens marine organisms that build calcium carbonate shells or skeletons, including "
        "corals, mollusks, and certain plankton species. Coral reefs, which support 25 percent "
        "of all marine species, are particularly vulnerable. Mass bleaching events have become "
        "five times more frequent since the 1980s. The loss of these ecosystems would devastate "
        "fisheries that feed hundreds of millions of people worldwide."
    ),
    (
        "Agricultural practices contribute approximately 10 to 12 percent of global greenhouse "
        "gas emissions directly, with additional indirect emissions from land use change. Livestock "
        "farming is the largest agricultural source, producing methane through enteric fermentation "
        "and nitrous oxide from manure management. Rice paddies generate significant methane "
        "emissions from anaerobic decomposition. Synthetic fertilizer production and application "
        "releases nitrous oxide, a greenhouse gas nearly 300 times more potent than CO2 over a "
        "100-year period. Sustainable agricultural practices including reduced tillage, cover "
        "cropping, agroforestry, and improved livestock management can significantly reduce these "
        "emissions while maintaining food production for a growing global population."
    ),
    (
        "Electric vehicles represent a transformative technology for reducing transportation "
        "emissions. Battery electric vehicles produce zero direct emissions and, depending on "
        "the electricity source, can reduce lifecycle emissions by 50 to 70 percent compared to "
        "conventional vehicles. Global EV sales have grown exponentially, reaching over 10 million "
        "units in 2022. Major automakers have committed to phasing out internal combustion engines "
        "within the next two decades. Challenges remain including battery cost, charging "
        "infrastructure availability, range limitations, and the environmental impact of lithium "
        "and cobalt mining. Public transportation electrification and cycling infrastructure "
        "development complement personal vehicle electrification in reducing overall transport "
        "sector emissions."
    ),
    (
        "Carbon capture utilization and storage technologies aim to remove CO2 from industrial "
        "emissions or directly from the atmosphere. Point-source capture at power plants and "
        "factories can prevent 85 to 95 percent of CO2 from reaching the atmosphere. Direct air "
        "capture facilities use chemical processes to extract CO2 from ambient air, though at "
        "significantly higher cost. Captured carbon can be stored in geological formations, used "
        "in enhanced oil recovery, or converted into useful products like building materials and "
        "synthetic fuels. Current global capture capacity is approximately 40 million tons per "
        "year, far below the gigatons needed to meet climate targets. Scaling these technologies "
        "requires substantial cost reductions and policy support."
    ),
    (
        "Biodiversity loss is accelerating as climate change alters habitats faster than many "
        "species can adapt. Current extinction rates are estimated at 100 to 1000 times the "
        "natural background rate. Warming temperatures force species to migrate toward poles or "
        "higher elevations, but many cannot move fast enough or face barriers like urban areas "
        "and oceans. Coral reef ecosystems face collapse from combined warming and acidification. "
        "Boreal forests are experiencing increased wildfire frequency and insect outbreaks. "
        "Pollinator populations are declining as flowering times shift out of sync with their "
        "life cycles. The loss of biodiversity undermines ecosystem services worth trillions of "
        "dollars annually, including pollination, water purification, and natural pest control."
    ),
    (
        "Sustainable urban planning plays a crucial role in climate change mitigation and "
        "adaptation. Cities account for over 70 percent of global CO2 emissions and house more "
        "than half the world population. Green building standards can reduce energy consumption "
        "by 30 to 50 percent through improved insulation, efficient heating and cooling systems, "
        "and smart energy management. Urban forests and green spaces help absorb CO2, reduce "
        "heat island effects, and manage stormwater. Compact mixed-use development reduces "
        "transportation needs while improving quality of life. Cities like Copenhagen, Singapore, "
        "and Portland have demonstrated that ambitious climate action can coexist with economic "
        "growth and improved livability for residents."
    ),
    (
        "The economic impacts of climate change are projected to be severe and unevenly "
        "distributed across regions and sectors. Without mitigation, global GDP could decline "
        "by 10 to 23 percent by 2100 compared to a world without climate change. Agricultural "
        "yields in tropical regions may fall by 20 to 30 percent, threatening food security for "
        "billions. Infrastructure damage from sea level rise, storms, and flooding could cost "
        "trillions of dollars. Health impacts including heat-related mortality, expanded disease "
        "vectors, and air pollution add further economic burden. However, the costs of climate "
        "action are estimated at 1 to 4 percent of global GDP, far less than the projected "
        "damages from inaction. Green investments also create economic opportunities and jobs."
    ),
    (
        "Methane emissions represent a critical near-term target for climate mitigation due to "
        "methane high global warming potential. Over a 20-year period, methane traps approximately "
        "80 times more heat than CO2. Major sources include fossil fuel extraction and processing, "
        "livestock agriculture, landfills, and wetlands. Thawing permafrost in Arctic regions "
        "releases stored methane, creating a dangerous positive feedback loop. The Global Methane "
        "Pledge launched in 2021 aims to reduce methane emissions by 30 percent below 2020 levels "
        "by 2030. Technologies for detecting and plugging methane leaks from oil and gas "
        "infrastructure have improved dramatically, making rapid reductions technically feasible "
        "and often economically beneficial."
    ),
    (
        "Climate adaptation strategies are essential alongside mitigation efforts as some degree "
        "of warming is already locked in. Coastal communities are investing in sea walls, managed "
        "retreat, and nature-based solutions like mangrove restoration. Agricultural adaptation "
        "includes developing drought-resistant crop varieties, improving irrigation efficiency, "
        "and diversifying farming systems. Urban areas are implementing heat action plans, "
        "expanding green infrastructure, and upgrading drainage systems. Water management "
        "strategies include rainwater harvesting, aquifer recharge, and desalination. Early "
        "warning systems for extreme weather events save lives and reduce economic losses. "
        "Adaptation finance remains insufficient, particularly for the most vulnerable nations "
        "that have contributed least to the problem."
    ),
    (
        "Scientific consensus on human-caused climate change is overwhelming, with 97 percent "
        "of actively publishing climate scientists agreeing that current warming is primarily "
        "driven by human activities. This consensus is supported by multiple independent lines "
        "of evidence including temperature records, ice cores, ocean measurements, and satellite "
        "observations. Climate models have successfully predicted warming trends observed over "
        "the past several decades. The IPCC Sixth Assessment Report synthesizes findings from "
        "thousands of peer-reviewed studies, confirming that it is unequivocal that human "
        "influence has warmed the atmosphere, ocean, and land. Remaining uncertainties relate "
        "primarily to the magnitude and timing of future changes, not whether human-caused "
        "warming is occurring."
    ),
    (
        "International climate finance mechanisms channel resources from developed to developing "
        "nations to support clean energy transitions and climate resilience. The Green Climate "
        "Fund, established under the UNFCCC, has mobilized over 10 billion dollars for climate "
        "projects in vulnerable countries. Multilateral development banks have increased climate "
        "lending to over 60 billion dollars annually. Carbon markets allow emission reductions "
        "to be traded, creating financial incentives for mitigation. However, the 100 billion "
        "dollar annual target promised by developed nations has not been consistently met. "
        "Innovative financing mechanisms including green bonds, climate insurance, and debt-for-"
        "nature swaps are expanding the toolkit for mobilizing private capital toward climate "
        "solutions."
    ),
    (
        "Youth climate movements have emerged as a powerful force for political change since "
        "2018. Inspired by activists like Greta Thunberg, millions of young people worldwide "
        "have participated in climate strikes demanding urgent government action. These movements "
        "have successfully shifted public discourse, influenced corporate policies, and pressured "
        "governments to declare climate emergencies. Legal challenges brought by youth plaintiffs "
        "have resulted in landmark court decisions requiring stronger climate action. Educational "
        "institutions are integrating climate literacy into curricula at all levels. The "
        "intergenerational equity argument that current generations have a moral obligation to "
        "protect the climate for future generations has gained significant legal and philosophical "
        "traction in policy debates worldwide."
    ),
]


def generate_mock_data(
    n_candidates: int = 200, seed: int = 42
) -> tuple[Query, list[Document]]:
    rng = random.Random(seed)
    query = Query(
        text="What are the causes and effects of climate change?",
        query_id="q1",
    )

    documents = []

    for i, passage in enumerate(PASSAGES):
        relevance = 0.95 - i * 0.012 + rng.uniform(-0.01, 0.01)
        relevance = max(0.3, relevance)
        token_count = len(passage.split())
        doc = Document(
            doc_id=f"d{i:03d}",
            text=passage,
            relevance_score=round(relevance, 4),
            token_count=token_count,
        )
        documents.append(doc)

    # Generate additional lower-relevance documents with partial concept overlap
    filler_templates = [
        "The study of {topic} involves understanding complex systems and their interactions "
        "with environmental factors including temperature precipitation and atmospheric "
        "composition over extended time periods using advanced computational models",
        "Research in {topic} has shown significant correlations between industrial development "
        "patterns and changes in regional weather systems affecting agricultural productivity "
        "and water resource availability in multiple geographic regions",
        "Analysis of {topic} data reveals important trends in ecosystem health and species "
        "distribution patterns that may be influenced by changing environmental conditions "
        "and human land use practices across different biomes",
        "The relationship between {topic} and global economic systems has been extensively "
        "studied with findings suggesting substantial impacts on trade patterns resource "
        "allocation and infrastructure investment decisions worldwide",
        "Recent advances in {topic} monitoring technology have enabled more precise "
        "measurements of atmospheric composition ocean temperature and ice sheet dynamics "
        "providing critical data for predictive modeling efforts",
    ]

    topics = [
        "atmospheric science", "marine ecology", "geological processes",
        "energy systems", "urban development", "agricultural science",
        "water resources", "forest ecology", "public health",
        "economic modeling", "policy analysis", "technology innovation",
        "social dynamics", "resource management", "environmental monitoring",
    ]

    for i in range(len(PASSAGES), n_candidates):
        template = rng.choice(filler_templates)
        topic = rng.choice(topics)
        text = template.format(topic=topic)
        # Add some random padding to vary length
        extra_words = rng.randint(0, 40)
        padding = " ".join(
            rng.choice("analysis research data model system process change impact factor".split())
            for _ in range(extra_words)
        )
        if padding:
            text = text + " " + padding

        relevance = 0.35 / (1 + 0.015 * (i - len(PASSAGES)))
        relevance += rng.uniform(-0.03, 0.03)
        relevance = max(0.05, round(relevance, 4))
        token_count = len(text.split())
        doc = Document(
            doc_id=f"d{i:03d}",
            text=text,
            relevance_score=relevance,
            token_count=token_count,
        )
        documents.append(doc)

    return query, documents
